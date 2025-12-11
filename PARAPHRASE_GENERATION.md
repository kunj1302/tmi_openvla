# Paraphrase Generation for LIBERO Evaluation

This document describes the prompt used to generate paraphrased task instructions for testing model robustness to natural language variations.

## Purpose

The paraphrased prompts are used to evaluate whether the Vision-Language-Action (VLA) model can handle natural language variations (synonyms, different sentence structures) without changing the core task. This is important for testing the model's robustness in real-world scenarios where users may phrase instructions differently.

## Prompt Used

The following prompt was used to generate the paraphrases in `experiments/object_paraphrased.json`:

---

**Role:** You are a Robotics Data Engineer generating a training dataset for a Vision-Language-Action (VLA) model like OpenVLA.

**Context:** I am testing a robot on the LIBERO benchmark. The robot must perform specific manipulation tasks (e.g., picking up an object and placing it in a basket). I need to test if the model can handle natural language variations (synonyms, different sentence structures) without changing the core task.

**Input Task List:**

1. "pick up the alphabet soup and place it in the basket"
2. "pick up the cream cheese and place it in the basket"
3. "pick up the salad dressing and place it in the basket"
4. "pick up the bbq sauce and place it in the basket"
5. "pick up the ketchup and place it in the basket"
6. "pick up the tomato sauce and place it in the basket"
7. "pick up the butter and place it in the basket"
8. "pick up the milk and place it in the basket"
9. "pick up the chocolate pudding and place it in the basket"
10. "pick up the orange juice and place it in the basket"

**Your Task:**

For EACH input instruction above, generate 5 "Semantic Paraphrases".

**Constraints:**

1. **Do not add irrelevant info.** Keep it concise.
2. **Vary the Verbs:** Use synonyms like "grab", "grasp", "lift", "move", "take".
3. **Vary the Structure:** Change the word order (e.g., "Move the soup to the basket" or "The basket needs the soup").
4. **Strict Object Permanence:** Do NOT change the identity of the object (e.g., do not change "alphabet soup" to "food"). The robot needs to recognize the specific object name to ground it visually.

**Output Format:**

Provide the output as a valid JSON object where the key is the original instruction and the value is a list of 5 paraphrased strings.

Example format:

```json
{
  "pick up the alphabet soup and place it in the basket": [
    "grab the alphabet soup and put it in the basket",
    "move the alphabet soup into the basket",
    ...
  ]
}
```

---

## Key Constraints

1. **Object Permanence**: The specific object name (e.g., "alphabet soup", "cream cheese") must remain unchanged. This is critical because the robot needs to visually ground the specific object in the scene.

2. **Verb Variation**: Paraphrases should use different action verbs while maintaining the same semantic meaning:
   - "pick up" → "grab", "grasp", "lift", "take"
   - "place" → "put", "move", "transfer", "deposit", "set"

3. **Structure Variation**: Paraphrases should vary sentence structure while keeping the instruction clear and actionable.

4. **Conciseness**: Paraphrases should remain concise and not add irrelevant information.

## Generated File

The generated paraphrases are stored in `experiments/object_paraphrased.json` and contain 5 paraphrased versions for each of the 10 LIBERO-Object tasks.

## Usage

See `LIBERO_SETUP.md` for instructions on how to use the paraphrase JSON file during evaluation.

