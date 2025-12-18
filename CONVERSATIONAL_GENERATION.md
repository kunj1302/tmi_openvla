# Conversational Prompt Generation for LIBERO Evaluation

This document describes the prompt used to generate conversational/fluff task instructions for testing model robustness to noisy, real-world language variations.

## Purpose

The conversational prompts are used to evaluate whether the Vision-Language-Action (VLA) model can handle natural language instructions that include:
- Extra commentary or context
- Conversational fillers
- Irrelevant information
- Longer, more natural phrasing

This is important for testing the model's robustness to real-world instructions where users may not provide perfectly concise commands.

## Prompt Used

The following prompt can be used to generate the conversational variants in `experiments/object_conversational.json`:

---

**Role:** You are a Robotics Data Engineer generating a training dataset for a Vision-Language-Action (VLA) model like OpenVLA.

**Context:** I am testing a robot on the LIBERO benchmark. The robot must perform specific manipulation tasks (e.g., picking up an object and placing it in a basket). I need to test if the model can handle conversational, noisy instructions that include extra commentary, context, or irrelevant information—simulating how real users might talk to a robot.

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

For EACH input instruction above, generate 5 "Conversational/Fluff" variants.

**Constraints:**

1. **Add conversational elements:** Include polite phrases, fillers, or context (e.g., "could you please...", "I need you to...", "carefully...", "make sure to...")
2. **Add extra commentary:** Include extra information that doesn't change the core task (e.g., "be careful not to knock anything over", "the basket is on the table", "try not to spill anything")
3. **Vary length:** Create instructions ranging from slightly longer to significantly more verbose
4. **Strict Object Permanence:** Do NOT change the identity of the object (e.g., do not change "alphabet soup" to "food" or "can"). The robot needs to recognize the specific object name.
5. **Keep core action intact:** The essential task (pick up object X, place in basket) must remain clear within the fluff

**Output Format:**

Provide the output as a valid JSON object where the key is the original instruction and the value is a list of 5 conversational variants.

Example format:

```json
{
  "pick up the alphabet soup and place it in the basket": [
    "hey, could you please grab the alphabet soup and carefully put it in the basket for me?",
    "I need you to pick up the alphabet soup and place it in the basket, but be careful not to knock anything over",
    "pick up the alphabet soup and move it to the basket - try to be gentle with it",
    "okay, so the basket is right there, and I need you to take the alphabet soup and put it inside",
    "can you help me by picking up that alphabet soup can and placing it into the basket? thanks!"
  ]
}