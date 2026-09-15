import os
import sys
import time
import shutil

# Get the project root folder
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Add project root to Python path
sys.path.insert(
    0,
    PROJECT_ROOT
)

from src.workflow_runner import run_workflow


# Folder settings
INBOX = "inbox"
OUTBOX = "outbox"
PROCESSED = "inbox/processed"
LOG_FILE = "logs/workflow.log"

# Check inbox every 3 seconds
POLL_SECONDS = 3


def main():

    # Create required folders
    os.makedirs(INBOX, exist_ok=True)
    os.makedirs(OUTBOX, exist_ok=True)
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Watching inbox for new .txt files...")
    print("Drop a .txt file into inbox/ to start processing.")
    print("Press Ctrl+C to stop.")

    while True:

        # Look through files in inbox
        for name in os.listdir(INBOX):

            # Only process .txt files
            if not name.lower().endswith(".txt"):
                continue

            path = os.path.join(INBOX, name)

            # Ignore folders
            if os.path.isdir(path):
                continue

            print(f"\nProcessing: {path}")

            try:

                # Run the complete workflow
                result = run_workflow(
                    path,
                    OUTBOX,
                    LOG_FILE
                )

                print(
                    f"Workflow completed: "
                    f"{result.get('ok')}"
                )

                # IMPORTANT:
                # Only move the file to processed
                # if the workflow actually succeeded.
                if result.get("ok"):

                    destination = os.path.join(
                        PROCESSED,
                        name
                    )

                    # Remove an older copy if it exists
                    if os.path.exists(destination):
                        os.remove(destination)

                    # Move successfully processed file
                    shutil.move(
                        path,
                        destination
                    )

                    print(
                        f"Moved to: {destination}"
                    )

                else:

                    # Keep failed files in inbox
                    # so they can be retried later.
                    print(
                        "Workflow failed. "
                        "File kept in inbox for retry."
                    )

            except Exception as error:

                print(
                    f"Workflow error: {error}"
                )

                # Do NOT move the file when
                # an unexpected error occurs.
                print(
                    "File kept in inbox for retry."
                )

        # Wait before checking again
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()