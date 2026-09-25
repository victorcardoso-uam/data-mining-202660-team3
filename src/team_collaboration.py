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
        },
        {
            "name": "Jose Carlos Verea Ovando",
            "student_id": "00473456",
            "role": "Data Engineer",
            "assigned_reviewer": "Carlos Emilio Mejia Martinez",
            "git_feature_branch": "feature/activity-10-jose",
            "preferred_ai_assistant": "ChatGPT",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "name": "Jimena Escalante Cámara",
            "student_id": "00473431",
            "role": "Data Analyst",
            "assigned_reviewer": "Victoria Morales Cabrera",
            "git_feature_branch": "feature/activity-10-jimena",
            "preferred_ai_assistant": "Claude AI",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "name": "Carlos Emilio Mejia Martinez",
            "student_id": "00473455",
            "role": "Data Scientist",
            "assigned_reviewer": "Jose Carlos Verea Ovando",
            "git_feature_branch": "feature/activity-10-carlos-emilio",
            "preferred_ai_assistant": "Bard AI",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
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