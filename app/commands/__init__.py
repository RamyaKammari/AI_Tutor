import logging


class Command:
    """Base class for all command plugins, with metadata for dynamic menu generation."""
    def __init__(self):
        self.name = ""  # Command name for menu display
        self.description = ""  # Command description for menu display

    def execute(self, *args, **kwargs):
        """Execute the command with given arguments."""
        raise NotImplementedError("Command execution not implemented.")

class CommandHandler:
    """Handles registration and execution of commands."""
    def __init__(self):
        self.commands = {}

    def register_command(self, command):
        """Register a command instance."""
        if command.name in self.commands:
            logging.warning(f"Command '{command.name}' is already registered. Overwriting.")
        self.commands[command.name] = command
        logging.info(f"Command '{command.name}' registered successfully.")
    
    def get_commands(self):
        """Return a list of command metadata for all registered commands."""
        return [(cmd.name, cmd.description) for cmd in self.commands.values()]

    def execute_command(self, name, *args):
        """Execute a command by name."""
        command = self.commands.get(name)
        if not command:
            logging.error(f"Command '{name}' not found.")
            return
        try:
            command.execute(*args)
        except Exception as e:
            logging.error(f"Error executing command '{name}': {e}")