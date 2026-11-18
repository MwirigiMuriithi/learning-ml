# ⚖️ Responsible-AI Checklist for an Agricultural Advisory Model

Answer each in 3–5 sentences in your notes. Weak answers are vague; strong answers name *who* is affected and *what you would do*.

## 1. Purpose & stakes
- [ ] What decision does the output influence? What is the cost of a false "healthy"? Of a false alarm? Does your threshold reflect that asymmetry?
- [ ] Could a farmer reasonably over-rely on the app? What does the UI say about its own limits?

## 2. Data
- [ ] Where did the images/prices come from? Were farmers informed and did they consent? Who owns the data and who benefits from the model?
- [ ] Which regions, varieties, seasons, phones and farm types are **missing**? How would performance differ for them?
- [ ] Are labels from experts or crowd/quick annotators? How noisy might they be (Lesson 14)?

## 3. Fairness & inclusion
- [ ] Who can't use this (no smartphone, low literacy, no connectivity, language)? Offline mode? Swahili/local-language advice? Voice or icon-based UI?
- [ ] Does accuracy differ across subgroups (devices, lighting, crop stages)? Did you measure it?

## 4. Uncertainty & honesty
- [ ] Does the system abstain when unsure? Are probabilities **calibrated** (check a reliability diagram)?
- [ ] Are intervals reported with their **measured coverage**? Do you state sample sizes and variance?

## 5. Robustness & monitoring
- [ ] What happens with blurry/dark/odd photos, a new disease, an unseen drought, a policy shock? How would you **detect** drift in production?
- [ ] Is there a feedback channel so extension officers can correct the model?

## 6. System effects
- [ ] If many farmers follow the same price advice, can it move the market? Could it be gamed?
- [ ] Who is accountable for wrong advice, and what is the escalation path?

## 7. Transparency
- [ ] Is there a model card? Can a non-expert understand what the model does and doesn't do?
