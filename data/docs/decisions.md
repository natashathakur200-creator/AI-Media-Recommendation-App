# MediaPilot AI — Key Design Decisions

## Why I chose this product and user

- MediaPilot AI was designed to help businesses plan their marketing and media activities using AI.
- The target users are businesses, particularly SMEs and marketing teams that need structured media planning support.
- The product combines business information, target audience, objectives, budget, and planning period to generate a practical media recommendation.

## How memory works

- The application uses session-based memory.
- Each conversation is associated with a unique `session_id`.
- Session data is stored in `data/sessions.json`.
- Each session stores business context, conversation messages, creation time, and update time.
- Conversation history is bounded by retrieving the most recent 10 messages.
- Sessions older than seven days can be cleaned up to prevent unlimited growth.

## What tools exist and how they are controlled

The application contains deterministic Python tools:

- File listing
- File reading
- File writing
- Document search
- Calculator

File operations are restricted to the defined workspace using path validation to prevent access outside the permitted directory.

The calculator uses an allow-list of supported arithmetic operations rather than executing arbitrary Python code.

## How automation is triggered

- The application contains a multi-step workflow runner.
- The workflow processes an input business/media request through multiple stages.
- It loads the input, extracts structured information, classifies the request, generates a media strategy, saves outputs, and records the workflow execution.
- Each stage checks whether the previous step succeeded before continuing.

## Validations and guardrails

The system applies multiple validation and safety mechanisms:

- Structured AI output is validated before downstream processing.
- Required fields are enforced.
- Urgency values are restricted to defined values.
- Empty or incomplete input is handled explicitly.
- User-provided business text is treated as untrusted data.
- Prompt-injection attempts are not followed.
- System instructions, developer instructions, API keys, and hidden information are not disclosed.
- The AI is instructed not to invent company facts or market statistics.
- Marketing budgets must be respected.
- ROI is presented as an estimate or scenario rather than a guaranteed result.
- Model quota errors are handled separately from other failures.

## Tradeoffs

- File-based persistence was chosen because it is simple to implement and suitable for a prototype or learning project.
- A database or distributed memory system would provide better scalability for a production environment.
- Deterministic Python tools were preferred over unrestricted code execution to improve control and safety.
- The architecture prioritizes transparency and simplicity over large-scale infrastructure.
- Gemini is used as the AI service while deterministic application logic handles validation, routing, persistence, and workflow execution.
