from datasets import Dataset
import numpy as np
import pickle
import hnswlib

from transformers import AutoTokenizer

def load_embeddings():
    # Load embeddings
    all_skill_embeddings = np.load("/home/ubuntu/vignav/data/open-r1/skills/skill_embeddings.npy")

    # Load skills list
    with open("/home/ubuntu/vignav/data/open-r1/skills/skills_list.pkl", "rb") as f:
        all_skills = pickle.load(f)

    task_embeddings = np.load("/home/ubuntu/vignav/data/open-r1/tasks/train_task_embeddings.npy")

    with open("/home/ubuntu/vignav/data/open-r1/tasks/train_tasks.pkl", "rb") as f:
        tasks = pickle.load(f)['prompts']
    
    for i, task in enumerate(tasks):
        t = task.split("You need to perform the following task for the user:")[-1].split("You have the following skills:")[0].strip()
        if t.endswith(".."):
            t = t[:-1]
        tasks[i] = t
    
    tasks = tasks[1000:] # used first 1000 for SFT
    task_embeddings = task_embeddings[1000:]

    idxs = []
    for i, skill in enumerate(all_skills):
        test_skills = ['set_cell_range_values', 'create_new_sheet', 'create_chart', 'group_columns', 'sort_range']
        for s in test_skills:
            if "Skill: " + s in skill:
                idxs.append(i)

    assert len(idxs) == len(test_skills)

    # create train_skills, train_skill_embeddings, test_skills, test_skill_embeddings
    skills = [all_skills[j] for j in range(len(all_skills)) if j not in idxs]
    skill_embeddings = np.array([all_skill_embeddings[j] for j in range(len(all_skill_embeddings)) if j not in idxs])
    # test_skills = [all_skills[j] for j in idxs]
    # test_skill_embeddings = np.array([all_skill_embeddings[j] for j in idxs])

    return skills, skill_embeddings, tasks, task_embeddings

def create_index(skill_embeddings):
    # Create and configure HNSW index
    dim = skill_embeddings.shape[1]  # dimensionality of embeddings
    num_elements = len(skill_embeddings)

    # Initialize index - the maximum number of elements should be known beforehand
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # Add items to index
    index.add_items(skill_embeddings)

    # Set ef parameter for search
    index.set_ef(50)  # higher ef leads to better accuracy but slower search

    return index

def search_skills(index, skills, query_embedding, k=5):
    """
    Get top k most similar items using cosine similarity
    Args:
        query_embedding: numpy array of shape (dim,)
        k: number of nearest neighbors to return
    Returns:
        indices of top k similar items and their distances
    """
    # Reshape query if needed
    if len(query_embedding.shape) == 2:
        query_embedding = query_embedding.reshape(-1)
        
    # Query the index
    labels, distances = index.knn_query(query_embedding, k=k)

    labels = [skills[idx] for idx in labels[0]]

    return labels[::-1], distances[0][::-1]

def generate_candidate_selector_prompt(candidate_skills: list, intent: str, tokenizer: AutoTokenizer) -> str:
    computer_use_skill = """Skill: fallback_computer_use

Description: Perform various computer interactions such as pressing keys, typing text, moving the mouse, clicking, dragging, and taking screenshots.

Args:
    action (str): The action to perform. The available actions are 'key', 'type', 'cursor_position', 'mouse_move', 'left_click', 'left_click_drag', 'right_click', 'middle_click', 'double_click', and 'screenshot'.
    coordinate (Optional[Tuple[int, int]]): A tuple (x, y) specifying the pixel coordinates for mouse actions.
        - Required for `mouse_move` and `left_click_drag`.
    text (Optional[str]): The text to type or the key to press.
        - Required for `type` and `key`.

Returns:
    Tuple[bool, str, Optional[Tuple[int, int]]]: A tuple containing:
        - bool: True if the operation was successful, False otherwise.
        - str: A success/error message describing the operation result.
        - Optional[Tuple[int, int]]: The cursor's (x, y) position, returned only when `action=cursor_position`.
"""
    candidate_skills = '\n'.join(candidate_skills + [computer_use_skill])

    prefix = [{
          "role": "system",
          "content": "You are a computer use agent using Microsoft Excel. You will be given an intent from the user, and you must match it to the most appropriate skill from a list of candidate skills. The skill you chopose will then be executed on the computer, and should accomplish the intent.\n\n.You will first reason through your decision and then provide the user with the appropriate skill name. All your thinking should be enclosed in <think></think> tags, you must then provide only the skill name in <answer>[SKILL_NAME]</answer> tags at the end of your response, e.g. if you decide the appropriate skill is `select_tab` your response will end with <answer>select_tab</answer>."
      },
      {
          "role": "user",
          "content": f"My intent is: {intent}\nSelect a skill from the following list that can accomplish this intent:\n\n{candidate_skills}"
      },
      {
          "role": "assistant",
          "content": "Let me solve this step by step.\n<think>"
      }]

    return {"prompt": tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)}

def load_action_dataset(model_name):
    """Load the action dataset."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    skills, skill_embeddings, tasks, task_embeddings = load_embeddings()
    index = create_index(skill_embeddings)
    
    training_samples = []

    for i, embed in enumerate(task_embeddings):
        intent = tasks[i]
        candidate_skills, _ = search_skills(index, skills, embed, k=10)  # Get top 10 skills
        
        # Generate prompt using your existing function
        prompt = generate_candidate_selector_prompt(candidate_skills, intent, tokenizer)['prompt']
        
        sample = {
            'prompt': prompt,
            'intent': intent,
            'candidates': candidate_skills
            }
        training_samples.append(sample)

        dataset = Dataset.from_list(training_samples)
        return dataset