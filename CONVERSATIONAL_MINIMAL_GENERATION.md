# Minimal Conversational Prompt Generation for LIBERO Evaluation

This document describes the prompt used to generate minimal conversational task instructions for testing model robustness to natural conversational language without excessive commentary.

## Purpose

The minimal conversational prompts are used to evaluate whether the Vision-Language-Action (VLA) model can handle natural language instructions that include:
- Conversational tone and politeness
- Natural phrasing variations
- Minimal extra context
- **Filtered excessive commentary** (compared to full conversational variants)

This is important for testing the model's robustness to realistic conversational instructions that maintain clarity while sounding natural.

## Prompt Used

The following prompt can be used to generate the minimal conversational variants in `experiments/object_conversational_minimal.json`:

---

**Role:** You are a Robotics Data Engineer generating a training dataset for a Vision-Language-Action (VLA) model like OpenVLA.

**Context:** I am testing a robot on the LIBERO benchmark. The robot must perform specific manipulation tasks (e.g., picking up an object and placing it in a basket). I need to test if the model can handle conversational instructions that sound natural and polite, but WITHOUT excessive commentary or irrelevant information.

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

For EACH input instruction above, generate 5 "Minimal Conversational" variants.

**Constraints:**

1. **Add conversational elements:** Include polite phrases or natural language (e.g., "could you please...", "can you...", "I need you to...")
2. **Keep it concise:** Avoid excessive commentary, warnings, or extra context that distracts from the core task
3. **Maintain clarity:** The instruction should be clear and focused on the task
4. **Natural phrasing:** Use variations that sound like natural human speech without being verbose
5. **Strict Object Permanence:** Do NOT change the identity of the object (e.g., do not change "alphabet soup" to "food" or "can"). The robot needs to recognize the specific object name.
6. **Keep core action intact:** The essential task (pick up object X, place in basket) must remain clear

**Examples of what to AVOID:**
- "be careful not to knock anything over"
- "the basket is on the table"
- "try not to spill anything"
- "make sure you're gentle"

**Examples of what to INCLUDE:**
- "could you please pick up the alphabet soup and place it in the basket?"
- "can you grab the alphabet soup and put it in the basket?"
- "I need you to pick up the alphabet soup and place it in the basket"
- "please pick up the alphabet soup and move it to the basket"

**Output Format:**

Provide the output as a valid JSON object where the key is the original instruction and the value is a list of 5 minimal conversational variants.

Example format:

```json
{
  "pick up the alphabet soup and place it in the basket": [
    "could you please pick up the alphabet soup and place it in the basket?",
    "can you grab the alphabet soup and put it in the basket?",
    "I need you to pick up the alphabet soup and place it in the basket",
    "please pick up the alphabet soup and move it to the basket",
    "can you help me by picking up the alphabet soup and placing it in the basket?"
  ]
}
```

---

## Usage

Use this prompt with a language model (e.g., GPT-4, Claude) to generate the variants, then save the output to `experiments/object_conversational_minimal.json`.

## Comparison with Other Variants

| Variant Type | Characteristics | Use Case |
|--------------|----------------|----------|
| **Original** | Concise, direct commands | Baseline performance |
| **Paraphrased** | Synonym substitutions, structural variations | Test linguistic flexibility |
| **Conversational (Full)** | Includes extra commentary, context, warnings | Test robustness to noise |
| **Conversational (Minimal)** | Natural tone without excessive fluff | Test realistic conversational input |

The minimal conversational variants aim to find a balance between natural human speech and clarity, representing how users might actually interact with a robot assistant in real scenarios.
