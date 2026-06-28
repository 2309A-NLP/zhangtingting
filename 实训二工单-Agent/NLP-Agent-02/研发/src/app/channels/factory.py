from app.channels.alert_base import ReminderAlertChannel
from app.channels.alert_console import ConsoleReminderAlertChannel
from app.channels.alert_webhook import WebhookReminderAlertChannel
from app.channels.base import ReminderChannel
from app.channels.console import ConsoleReminderChannel
from app.channels.webhook import WebhookReminderChannel
from app.core.config import settings


def get_reminder_channel() -> ReminderChannel:
    if settings.reminder_channel == "console":
        return ConsoleReminderChannel()
    if settings.reminder_channel == "webhook":
        return WebhookReminderChannel()
    return ConsoleReminderChannel()


def get_reminder_alert_channel() -> ReminderAlertChannel:
    if settings.reminder_alert_channel == "webhook":
        return WebhookReminderAlertChannel()
    return ConsoleReminderAlertChannel()
