# MediaPilot AI — Demo Script

## Demo Duration

Target duration: 3–5 minutes

## 1. Problem — 10 seconds

"Businesses often know they need marketing, but deciding where to spend their budget, which channels to use, and how to measure expected outcomes can be difficult.

MediaPilot AI helps turn business information and marketing objectives into a structured media planning recommendation."

## 2. Product Overview — 20 seconds

"MediaPilot AI is an AI-powered media planning application.

A user provides information about their business, target audience, marketing objective, budget, and planning period.

The system then generates a recommended media strategy, media mix, execution approach, ROI scenarios, and supporting evidence."

## 3. Architecture — 30 seconds

"Behind the interface, the application uses a FastAPI backend connected to Google Gemini.

The system also includes persistent session memory, deterministic tools, a multi-step workflow engine, and validation guardrails.

The workflow can extract structured information, classify the request, generate a media strategy, save outputs, and log execution."

[Show `docs/architecture.md` architecture diagram]

## 4. Live Demo — Approximately 2 minutes

### Step 1 — Create a Media Plan

Open the deployed application:

https://ai-media-recommendation-app.onrender.com/

Enter a realistic business example.

For example:

- Industry: Solar & Renewable Energy
- Business Type: B2B2C
- Location: Jaipur, Rajasthan
- Target Audience: Residential and commercial customers
- Marketing Objective: Lead generation and brand awareness
- Budget: ₹5,00,000
- Planning Period: 12 months

Click **Generate Media Plan**.

### Step 2 — Show the Recommendation

Explain the generated result:

"The system converts these inputs into a structured media plan.

Here we can see the strategic direction, recommended budget allocation, media mix, execution plan, ROI scenarios, and evidence."

Show the different result sections/tabs.

### Step 3 — Demonstrate the AI Planning Assistant

Open the AI Planning Assistant.

Ask a follow-up question such as:

"Why did you recommend this media mix for my business?"

Then ask:

"How can I improve the allocation if my main objective is lead generation?"

Explain:

"The assistant can use the conversation context to provide follow-up guidance rather than treating every question as a completely new conversation."

## 5. Tool + Workflow — 30 seconds

"Alongside the user-facing recommendation system, the project includes deterministic tools and a multi-step workflow engine."

Show the GitHub `src/tools.py` file.

Briefly explain:

"The tools include controlled file operations, document search, and a calculator. File access is restricted to the permitted workspace."

Then show:

`src/workflow_runner.py`

Say:

"The workflow chains multiple stages: loading input, extracting structured information, classification and routing, media strategy generation, saving outputs, and logging."

## 6. Guardrails & Evaluation — 20 seconds

Show the validation/guardrail implementation.

Say:

"The system doesn't blindly trust model output.

Structured responses are validated before they continue through the workflow. The application also treats user-provided text as untrusted input, blocks prompt-injection instructions, avoids fabricated business facts and market statistics, respects the stated budget, and does not guarantee ROI."

## 7. Deployment & Closing — 10 seconds

"The application is deployed publicly on Render and can be accessed through this URL.

This project demonstrates how an AI application can combine an LLM with memory, deterministic tools, workflow automation, validation, and a real user interface."

Live URL:

https://ai-media-recommendation-app.onrender.com/
