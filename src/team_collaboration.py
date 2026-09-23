"""
Team Collaboration & Engineering Contribution Registry
Data Mining & Modern AI Systems (IIND4417) - Session 13
"""

import datetime


TEAM_REGISTRY = {
    "cohort": "Team 3",
    "repository": "data-mining-202660-team3",
    "members": [
        {
            "name": "Jimena Escalante",
            "student_id": "00470871",
            "role": "Data Analyst",
            "assigned_reviewer": "María Jimena Escalante Cámara",
            "git_feature_branch": "feature/activity-10-jimena-escalante",
            "preferred_ai_assistant": "ChatGPT",
            "timestamp": datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
    ],
}


def display_team_roster():
    print(
        f"\n{'=' * 20} "
        f"{TEAM_REGISTRY['cohort']} ACTIVE ROSTER "
        f"{'=' * 20}"
    )

    for member in TEAM_REGISTRY["members"]:
        print(
            f"* {member['name']} "
            f"({member['student_id']}) | "
            f"Role: {member['role']} | "
            f"Branch: {member['git_feature_branch']}"
        )

    print("=" * 60)


if __name__ == "__main__":
    display_team_roster()