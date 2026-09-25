"""Team collaboration registry for Activity 10."""

import datetime


TEAM_REGISTRY = {
    "cohort": "Team 3",
    "repository": "data-mining-202660-team3",
    "members": [
        {
            "name": "EDUARDO GALLEGOS BOLAÑOS CACHO",
            "student_id": "00474516",
            "role": "AI DATA ENGINEER",
            "assigned_reviewer": "JIMENA ESCALANTE CÁMARA",
            "git_feature_branch": "feature/activity-10-eduardo-gallegos",
            "preferred_ai_assistant": "Codex",
            "timestamp": datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        },
    ],
}


def display_team_roster():
    """Display each registered member."""
    print(f"\n{'=' * 20} {TEAM_REGISTRY['cohort']} ACTIVE ROSTER {'=' * 20}")
    for member in TEAM_REGISTRY["members"]:
        print(
            f"* {member['name']} ({member['student_id']}) | "
            f"Role: {member['role']} | "
            f"Branch: {member['git_feature_branch']}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    display_team_roster()
