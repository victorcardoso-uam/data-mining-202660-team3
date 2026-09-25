"""
Team Collaboration & Engineering Contribution Registry
Data Mining & Modern AI Systems (IIND4417) — Session 13
"""

import datetime


TEAM_REGISTRY = {
    "cohort": "Team 3",
    "repository": "data-mining-202660-team3",
    "members": [
        {
            "name": "Victoria Morales Cabrera",
            "student_id": "473430",
            "role": "Data Quality Auditor",
            "assigned_reviewer": "Jimena Escalante Cámara",
            "git_feature_branch": "feature/activity-10-victoria-morales",
            "preferred_ai_assistant": "Perplexity AI",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    ],
}


def display_team_roster():
    """Display the current registered team members."""
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