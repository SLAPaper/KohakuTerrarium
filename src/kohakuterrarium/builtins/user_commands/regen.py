"""Regenerate the last assistant response with the current agent settings."""

from kohakuterrarium.builtins.user_commands.registry import register_user_command
from kohakuterrarium.core.interrupt_regeneration import RegenerationTurnError
from kohakuterrarium.modules.user_command.base import (
    BaseUserCommand,
    CommandLayer,
    UserCommandContext,
    UserCommandResult,
)


@register_user_command("regen")
class RegenCommand(BaseUserCommand):
    name = "regen"
    aliases = ["regenerate", "retry"]
    description = "Regenerate the last assistant response with current settings"
    layer = CommandLayer.AGENT

    async def _execute(
        self, args: str, context: UserCommandContext
    ) -> UserCommandResult:
        if not context.agent:
            return UserCommandResult(error="No agent context.")
        try:
            receipt = await context.agent.regenerate_last_response_result()
        except RegenerationTurnError as exc:
            return UserCommandResult(
                error=str(exc),
                data={"operation": exc.receipt, "applied": exc.may_have_run},
            )
        return UserCommandResult(
            output="Regenerated last response.",
            data={"operation": receipt, "applied": True},
        )
