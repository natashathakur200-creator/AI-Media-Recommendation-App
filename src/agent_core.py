import os
from typing import Any, Dict, List, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src import tools as tool_impl


ToolFn = Callable[..., Dict[str, Any]]


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY: Dict[str, ToolFn] = {
    "list_files": tool_impl.list_files,
    "read_file": tool_impl.read_file,
    "write_file": tool_impl.write_file,
    "search_docs": tool_impl.search_docs,
    "calculator": tool_impl.calculator,
}


# ============================================================
# TOOL DECLARATIONS
# ============================================================

TOOL_DECLARATIONS = [

    types.FunctionDeclaration(
        name="list_files",
        description=(
            "List files inside the workspace directory. "
            "Use this to discover files created by the agent."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "path": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Workspace-relative folder path. "
                        "Use an empty string for the workspace root."
                    ),
                )
            },
        ),
    ),

    types.FunctionDeclaration(
        name="read_file",
        description=(
            "Read a text file from the workspace directory."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "path": types.Schema(
                    type=types.Type.STRING,
                    description="Workspace-relative file path.",
                )
            },
            required=["path"],
        ),
    ),

    types.FunctionDeclaration(
        name="write_file",
        description=(
            "Write a text file to the workspace directory."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "path": types.Schema(
                    type=types.Type.STRING,
                    description="Workspace-relative file path.",
                ),
                "content": types.Schema(
                    type=types.Type.STRING,
                    description="The complete file contents.",
                ),
            },
            required=["path", "content"],
        ),
    ),

    types.FunctionDeclaration(
        name="search_docs",
        description=(
            "Search local .txt documents inside data/docs "
            "and return matching snippets."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="Text to search for.",
                ),
                "max_hits": types.Schema(
                    type=types.Type.INTEGER,
                    description="Maximum number of matching documents.",
                ),
            },
            required=["query"],
        ),
    ),

    types.FunctionDeclaration(
        name="calculator",
        description=(
            "Safely calculate a basic arithmetic expression. "
            "Example: 12 * 49.99"
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "expression": types.Schema(
                    type=types.Type.STRING,
                    description="Arithmetic expression to calculate.",
                )
            },
            required=["expression"],
        ),
    ),
]


AGENT_TOOL = types.Tool(
    function_declarations=TOOL_DECLARATIONS
)


# ============================================================
# AGENT LOOP
# ============================================================

def run_agent(
    goal: str,
    max_steps: int = 8
) -> Dict[str, Any]:

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY in .env"
        )

    client = genai.Client(
        api_key=api_key
    )


    # ========================================================
    # SYSTEM INSTRUCTION
    # ========================================================

    system_instruction = """
You are a task-executing AI media planning agent.

You can use tools to:

- search local project documents
- read files
- write files inside the workspace
- perform calculations

Your job is to complete the user's goal by using tools when necessary.

Important rules:

- Only write files inside the workspace directory.
- Use tools when they are needed to complete the task.
- Do not invent information from files.
- Keep the number of steps bounded.
- Stop when the task is complete.
- If the goal is genuinely ambiguous, ask one clarifying question.
- For calculations, use the calculator tool rather than guessing.
- For file creation, actually use write_file.
- After successfully completing the requested task, stop using tools
  and provide a concise final answer.
"""


    # ========================================================
    # INITIAL CONTENT
    # ========================================================

    contents: List[Any] = [

        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"GOAL:\n{goal}"
                )
            ],
        )

    ]


    # ========================================================
    # AGENT LOOP
    # ========================================================

    for step in range(
        1,
        max_steps + 1
    ):

        print(
            f"\n--- Agent Step {step} ---"
        )


        # ====================================================
        # ASK GEMINI WHAT TO DO
        # ====================================================

        response = client.models.generate_content(

            model="gemini-3.6-flash",

            contents=contents,

            config=types.GenerateContentConfig(

                system_instruction=system_instruction,

                tools=[AGENT_TOOL],

                temperature=0.2,
            ),
        )


        # ====================================================
        # SHOW DEBUG INFORMATION
        # ====================================================

        if response.function_calls:

            print(
                "Gemini requested tool(s):"
            )

            for function_call in response.function_calls:

                print(
                    f"  - {function_call.name}"
                )

                print(
                    f"    Arguments: "
                    f"{dict(function_call.args) if function_call.args else {}}"
                )

        else:

            print(
                "Gemini returned a final response."
            )


        # ====================================================
        # NO TOOL CALL = FINAL ANSWER
        # ====================================================

        if not response.function_calls:

            final_text = (
                response.text or ""
            ).strip()

            return {
                "ok": True,
                "steps": step,
                "final": final_text,
            }


        # ====================================================
        # SAVE GEMINI'S TOOL REQUEST
        # ====================================================

        contents.append(
            response.candidates[0].content
        )


        # ====================================================
        # EXECUTE TOOLS
        # ====================================================

        tool_response_parts = []


        for function_call in response.function_calls:

            name = function_call.name

            args = (
                dict(function_call.args)
                if function_call.args
                else {}
            )


            print(
                f"Executing tool: {name}"
            )


            # ------------------------------------------------
            # CHECK TOOL
            # ------------------------------------------------

            if name not in TOOL_REGISTRY:

                tool_result = {
                    "ok": False,
                    "error": (
                        f"Unknown tool: {name}"
                    ),
                }


            # ------------------------------------------------
            # EXECUTE TOOL
            # ------------------------------------------------

            else:

                try:

                    tool_result = TOOL_REGISTRY[name](
                        **args
                    )

                except Exception as error:

                    tool_result = {
                        "ok": False,
                        "error": str(error),
                    }


            # ------------------------------------------------
            # SHOW TOOL RESULT
            # ------------------------------------------------

            print(
                f"Tool result: {tool_result}"
            )


            # ------------------------------------------------
            # CREATE FUNCTION RESPONSE
            # ------------------------------------------------

            tool_response_parts.append(

                types.Part.from_function_response(

                    name=name,

                    response=tool_result,

                )

            )


        # ====================================================
        # SEND TOOL RESULTS BACK TO GEMINI
        #
        # Gemini does not accept role="tool" in this setup.
        # Therefore we use role="user".
        # ====================================================

        contents.append(

            types.Content(

                role="user",

                parts=tool_response_parts,

            )

        )


    # ========================================================
    # MAX STEPS REACHED
    # ========================================================

    return {

        "ok": False,

        "error": (
            f"Max steps exceeded ({max_steps}). "
            "Try a narrower goal."
        ),

    }