"""Turn a student's own inputs and score into study advice.

The rules read the values that were submitted. They do not look up a fixed
paragraph for every student. Thresholds for absences come from the dataset
median and 75th percentile when those numbers are supplied.
"""

from src.config import STUDY_TIME_LABELS, category_for


def build_recommendations(
    inputs: dict,
    predicted_score: float,
    category: str,
    stats: dict,
) -> list[dict]:
    """Return up to six advice cards, highest priority first."""
    study = int(inputs["studytime"])
    failures = int(inputs["failures"])
    absences = int(inputs["absences"])
    median_absences = float(stats["absences_median"])
    high_absences = float(stats["absences_p75"])
    advice = []

    if study == 1:
        advice.append(
            {
                "priority": "High",
                "title": "Increase weekly study time",
                "detail": (
                    "Study time is under 2 hours per week, the lowest band in this "
                    "dataset. Add three 45-minute sessions on weekdays and one "
                    "90-minute weekend session so the week reaches about 5 hours."
                ),
            }
        )
    elif study == 2 and predicted_score < 16:
        advice.append(
            {
                "priority": "Medium",
                "title": "Add one focused study block",
                "detail": (
                    "Study time is 2 to 5 hours per week. Keep that habit and add "
                    "one 45-minute block for the hardest topic, using practice "
                    "questions rather than rereading notes."
                ),
            }
        )
    elif study >= 3 and predicted_score < 10:
        advice.append(
            {
                "priority": "Medium",
                "title": "Change the study method, not just the hours",
                "detail": (
                    f"Weekly study time is already {STUDY_TIME_LABELS[study]}, but "
                    "the predicted grade is still below the pass mark. Swap passive "
                    "reading for timed questions and a weekly teacher check."
                ),
            }
        )
    else:
        advice.append(
            {
                "priority": "Low",
                "title": "Protect the current study habit",
                "detail": (
                    f"Weekly study time is {STUDY_TIME_LABELS[study]}. That is a "
                    "strength. Keep the same days during test weeks, and spend the "
                    "sessions on practice questions."
                ),
            }
        )

    if failures >= 2:
        advice.append(
            {
                "priority": "High",
                "title": "Repair earlier course failures first",
                "detail": (
                    f"Past class failures are recorded as {failures}. Start each "
                    "study session with the topics from those failures, and ask a "
                    "teacher to check one problem set this week."
                ),
            }
        )
    elif failures == 1:
        advice.append(
            {
                "priority": "Medium",
                "title": "Revisit the failed topic before it compounds",
                "detail": (
                    "There is 1 past class failure. List the question types missed "
                    "in that course and put them at the start of the next two study "
                    "sessions."
                ),
            }
        )

    absence_cutoff = max(high_absences, 8)
    if absences >= absence_cutoff:
        advice.append(
            {
                "priority": "High",
                "title": "Bring absences down",
                "detail": (
                    f"{absences} absences is at or above the dataset's high band "
                    f"({absence_cutoff:.0f}, based on the 75th percentile and a "
                    "floor of 8). Missing class removes the explanations these "
                    "grades depend on. Set a weekly absence limit, and do one "
                    "worked example from a missed lesson on the same day."
                ),
            }
        )
    elif absences > median_absences:
        advice.append(
            {
                "priority": "Medium",
                "title": "Absences are above the class median",
                "detail": (
                    f"{absences} absences is above the dataset median of "
                    f"{median_absences:.0f}. Keep the next two weeks at or below "
                    "that median before adding anything else to the schedule."
                ),
            }
        )
    else:
        advice.append(
            {
                "priority": "Low",
                "title": "Attendance is a strength",
                "detail": (
                    f"{absences} absences is at or below the dataset median of "
                    f"{median_absences:.0f}. Keep that pattern in the weeks before "
                    "the final grade is set."
                ),
            }
        )

    if inputs["internet"] == "no":
        advice.append(
            {
                "priority": "Medium",
                "title": "Plan for study without home internet",
                "detail": (
                    "Internet at home is marked no. Download the week's practice "
                    "at school, or book a library computer on fixed days, so study "
                    "time does not depend on a connection."
                ),
            }
        )
    elif study <= 2:
        advice.append(
            {
                "priority": "Medium",
                "title": "Use home internet for one fixed practice slot",
                "detail": (
                    "Home internet is available and study time is still in a low "
                    "band. Choose one weekday evening for timed practice and close "
                    "everything else during that slot."
                ),
            }
        )

    if inputs["schoolsup"] == "no" and (failures >= 1 or predicted_score < 12):
        advice.append(
            {
                "priority": "High",
                "title": "Ask for school support",
                "detail": (
                    "Extra school support is marked no, and this record shows a "
                    "grade risk. Ask this week what support exists for the course "
                    "and bring one specific topic to the first session."
                ),
            }
        )
    elif inputs["schoolsup"] == "yes":
        advice.append(
            {
                "priority": "Low",
                "title": "Bring a question to school support",
                "detail": (
                    "School support is marked yes. Take one written question to "
                    "that session each week so the time stays tied to this course."
                ),
            }
        )

    if inputs["famsup"] == "no" and predicted_score < 14:
        advice.append(
            {
                "priority": "Medium",
                "title": "Add a ten-minute family check-in",
                "detail": (
                    "Family educational support is marked no. A short weekly "
                    "check-in with a parent or guardian about the next deadline "
                    "makes the study plan harder to skip."
                ),
            }
        )

    if inputs["activities"] == "yes" and absences >= absence_cutoff:
        advice.append(
            {
                "priority": "Medium",
                "title": "Check whether activities are crowding class",
                "detail": (
                    "Extra activities are yes and absences are in the high band. "
                    "If practices are the reason for missed lessons, pause the "
                    "activity until attendance is back under the median."
                ),
            }
        )
    elif inputs["activities"] == "no" and study <= 2 and predicted_score < 14:
        advice.append(
            {
                "priority": "Low",
                "title": "Leave extra activities for later",
                "detail": (
                    "Extra activities are marked no. Do not add one until weekly "
                    "study time is in the 5-to-10-hour band. The grade comes first."
                ),
            }
        )

    if inputs["higher"] == "yes" and predicted_score < 10:
        advice.append(
            {
                "priority": "High",
                "title": "Reach a pass before aiming further",
                "detail": (
                    "Higher education is the plan, and the predicted grade is "
                    "below 10 out of 20. On this scale, 10 is the usual pass mark. "
                    "The next target is a pass, then 14."
                ),
            }
        )
    elif inputs["higher"] == "yes" and predicted_score >= 16:
        advice.append(
            {
                "priority": "Low",
                "title": "The predicted grade supports the higher-education plan",
                "detail": (
                    f"Higher education is marked yes and the predicted grade is "
                    f"{predicted_score:.2f}. Protect attendance and the current "
                    "study schedule rather than adding more hours."
                ),
            }
        )
    elif inputs["higher"] == "no" and predicted_score >= 14:
        advice.append(
            {
                "priority": "Low",
                "title": "The predicted grade keeps later options open",
                "detail": (
                    "Higher education is marked no, and the predicted grade is "
                    "still in the good range. That leaves room to change the plan "
                    "later without first repairing the grade."
                ),
            }
        )

    if inputs.get("use_priors") and inputs.get("G1") is not None and inputs.get("G2") is not None:
        first_grade = float(inputs["G1"])
        second_grade = float(inputs["G2"])
        if second_grade + 1 <= first_grade:
            advice.append(
                {
                    "priority": "High",
                    "title": "Stop the slide between period grades",
                    "detail": (
                        f"Period 1 was {first_grade:.0f} and period 2 was "
                        f"{second_grade:.0f}. The grade is moving down. List the "
                        "question types missed in period 2 and practice those "
                        "before the final."
                    ),
                }
            )
        elif second_grade >= first_grade + 1:
            advice.append(
                {
                    "priority": "Low",
                    "title": "Keep the gain from period 1 to period 2",
                    "detail": (
                        f"The grade moved from {first_grade:.0f} to "
                        f"{second_grade:.0f}. Repeat the study pattern used before "
                        "period 2 instead of changing it before the final."
                    ),
                }
            )

    if inputs["address"] == "R" and absences > median_absences:
        advice.append(
            {
                "priority": "Medium",
                "title": "Plan around the rural commute",
                "detail": (
                    "The address type is rural and absences are above the dataset "
                    "median. Pick the days travel is most likely to fail and move "
                    "that lesson's practice to the previous evening."
                ),
            }
        )

    if (
        int(inputs["Medu"]) <= 1
        and int(inputs["Fedu"]) <= 1
        and inputs["schoolsup"] == "no"
    ):
        advice.append(
            {
                "priority": "Medium",
                "title": "Use school staff as the main academic support",
                "detail": (
                    "Both parents are recorded at primary education or none, and "
                    "school support is not in place. That is a reason to ask "
                    "teachers for subject help, not a limit on the grade."
                ),
            }
        )

    if category == "Needs Improvement":
        advice.append(
            {
                "priority": "High",
                "title": "Aim at 10 out of 20 first",
                "detail": (
                    f"The predicted final grade is {predicted_score:.2f}, in Needs "
                    "Improvement. Choose one action above and do it this week. The "
                    "practical target is the pass mark of 10, then a move toward 14."
                ),
            }
        )
    elif category == "Average":
        advice.append(
            {
                "priority": "Medium",
                "title": "Move a pass toward 14",
                "detail": (
                    f"The predicted final grade is {predicted_score:.2f}, in Average. "
                    "A pass is in reach. One timed past paper each weekend is the "
                    "most direct step toward 14."
                ),
            }
        )
    elif category == "Good":
        advice.append(
            {
                "priority": "Low",
                "title": "Close the gap to 16",
                "detail": (
                    f"The predicted final grade is {predicted_score:.2f}, in Good. "
                    "Excellent on this scale starts at 16. Review only the questions "
                    "missed on the last paper, not the whole notebook."
                ),
            }
        )
    else:
        advice.append(
            {
                "priority": "Low",
                "title": "Hold the routine that produced this range",
                "detail": (
                    f"The predicted final grade is {predicted_score:.2f}, in "
                    "Excellent (16 to 20). The main risk is a late rise in absences "
                    "or a drop in study time."
                ),
            }
        )

    order = {"High": 0, "Medium": 1, "Low": 2}
    advice.sort(key=lambda item: order[item["priority"]])
    unique = []
    seen = set()
    for item in advice:
        if item["title"] in seen:
            continue
        seen.add(item["title"])
        unique.append(item)
    return unique[:6]


def study_suggestions(inputs: dict, predicted_score: float) -> list[str]:
    """Concrete session plan based on the submitted study time, failures, and score."""
    study = int(inputs["studytime"])
    suggestions = []

    if study == 1:
        suggestions.append(
            "This week, study on Monday, Wednesday, and Friday for 45 minutes, plus 90 minutes on Saturday. Stop when the timer ends."
        )
    elif study == 2:
        suggestions.append(
            "Keep the current 2 to 5 hours, and add one 40-minute session that is only past-paper questions."
        )
    elif study == 3:
        suggestions.append(
            "Split the 5 to 10 hours into blocks of 45 minutes. End each block by writing three questions you still cannot answer."
        )
    else:
        suggestions.append(
            "Study time is already above 10 hours. Replace extra rereading with timed questions so the extra hours stay useful."
        )

    if int(inputs["failures"]) > 0:
        suggestions.append(
            f"Because there are {int(inputs['failures'])} past failures, start every session with 20 minutes on a failed topic before new material."
        )
    else:
        suggestions.append(
            "There are no past failures on the form. Start each session with a 10-question quiz from the last two weeks, then study the misses."
        )

    if int(inputs["absences"]) >= 5:
        suggestions.append(
            f"After any of the {int(inputs['absences'])} absences already on record, write the missed lesson title and complete one worked example the same day."
        )
    else:
        suggestions.append(
            "Once a week, redo a problem you got wrong in class without looking at the notes."
        )

    if predicted_score < 10:
        suggestions.append(
            "Work toward a pass: ask the teacher for the five highest-weight topics and stay with those until they are solid."
        )
    elif predicted_score < 14:
        suggestions.append(
            "To move from a pass toward 14, complete one full past test under timed conditions each weekend."
        )
    else:
        suggestions.append(
            "Before the final, take one timed paper and review only the questions you missed."
        )

    # category_for is applied by the caller; this check keeps the plan aligned if a score sits on a boundary.
    if category_for(predicted_score) == "Needs Improvement" and study >= 3:
        suggestions.append(
            "The hours are already high for a score in this band. Change the method: timed questions, then a teacher check, rather than more passive reading."
        )

    return suggestions[:4]
