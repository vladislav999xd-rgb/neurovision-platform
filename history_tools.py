"""Необязательный общий помощник. Предметные условия пишет команда."""
from copy import deepcopy


def append_event(state, action, record_id, field, before, after,
                 related_ids, details):
    event_id = 1
    for previous in state["history"]:
        if previous["event_id"] >= event_id:
            event_id = previous["event_id"] + 1
    event = {"event_id": event_id, "action": action,
             "record_id": record_id, "field": field,
             "before": deepcopy(before), "after": deepcopy(after),
             "related_ids": deepcopy(related_ids),
             "details": deepcopy(details)}
    state["history"].append(event)
    return event


def events_for_record(state, record_id):
    result = []
    for event in state["history"]:
        if event["record_id"] == record_id or record_id in event["related_ids"]:
            result.append(event)
    return result

