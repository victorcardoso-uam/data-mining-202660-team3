# File: src/team_collaboration.py
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
            "name": "Rodrigo Meouchi",
            "student_id": "00480093",
            "role": "Data Quality Auditor",
            "assigned_reviewer": "Victor Cardoso",
            "git_feature_branch": "feature/activity-10-rodrigo-meouchi",
            "preferred_ai_assistant": "Claude",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        # Teammates append their dictionary blocks via their own branches
    ],
}


def display_team_roster():
    print(f"\n{'=' * 20} {TEAM_REGISTRY['cohort']} ACTIVE ROSTER {'=' * 20}")
    for m in TEAM_REGISTRY["members"]:
        print(
            f"* {m['name']} ({m['student_id']}) | Role: {m['role']} "
            f"| Branch: {m['git_feature_branch']}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    display_team_roster()

    
    
