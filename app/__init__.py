import os
import pkgutil
import importlib
import sys
import logging.config
from dotenv import load_dotenv

from app.commands import Command, CommandHandler


class App:
    def __init__(self):
        os.makedirs('logs', exist_ok=True)
        self.configure_logging()
        load_dotenv()
        self.settings = self.load_environment_variables()
        self.command_handler = CommandHandler()

    def configure_logging(self):
        logging_conf_path = 'logging.conf'
        if os.path.exists(logging_conf_path):
            logging.config.fileConfig(logging_conf_path, disable_existing_loggers=False)
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Logging configured.")

    def load_environment_variables(self):
        settings = {key: value for key, value in os.environ.items()}
        logging.info("Environment variables loaded.")
        return settings

    def load_plugins(self):
        plugins_package = 'app.plugins'
        plugins_path = plugins_package.replace('.', '/')
        if not os.path.exists(plugins_path):
            logging.warning(f"Plugins directory '{plugins_path}' not found.")
            return
        for _, plugin_name, is_pkg in pkgutil.iter_modules([plugins_path]):
            if is_pkg:
                try:
                    plugin_module = importlib.import_module(f'{plugins_package}.{plugin_name}')
                    self.register_plugin_commands(plugin_module)
                except ImportError as e:
                    logging.error(f"Error importing plugin {plugin_name}: {e}")

    def register_plugin_commands(self, plugin_module):
        for item_name in dir(plugin_module):
            item = getattr(plugin_module, item_name)
            if isinstance(item, type) and issubclass(item, Command) and item is not Command:
                command_instance = item()
                self.command_handler.register_command(command_instance)
                logging.info(f"Command '{command_instance.name}' from plugin '{command_instance.name}' registered.")

    def start(self):
        self.load_plugins()
        # Dynamically generate and register the menu command
        dynamic_menu_command = DynamicMenuCommand(self.command_handler)
        self.command_handler.register_command(dynamic_menu_command)
        
        logging.info("Application started. Type 'show_menu' to see the menu or 'exit' to exit.")
        try:
            while True:
                cmd_input = input(">>> ").strip()
                if cmd_input.lower() == 'exit':
                    logging.info("Application exit.")
                    break  # Graceful exit without sys.exit(0) for more natural control flow in some contexts
                elif cmd_input == '':  # Check if the input is empty
                    self.command_handler.execute_command("show_menu")  # Execute the show_menu command
                else:
                    try:
                        cmd_name, *args = cmd_input.split()
                        self.command_handler.execute_command(cmd_name, *args)
                    except KeyError:
                        logging.error(f"Unknown command: {cmd_input}")
                        self.command_handler.execute_command("show_menu")  # Show menu if unknown command
                    except Exception as e:
                        logging.error(f"Error executing command: {e}")
        except KeyboardInterrupt:
            logging.info("Application interrupted and exiting gracefully.")
        finally:
            logging.info("Application shutdown.")


class DynamicMenuCommand(Command):
    def __init__(self, command_handler):
        super().__init__()
        self.name = "show_menu"
        self.description = "Show the dynamic menu of all commands."
        self.command_handler = command_handler

    def execute(self, *args, **kwargs):
        commands = self.command_handler.get_commands()
        menu = "Application Menu:\n"
        for name, description in commands:
            menu += f"{name}: {description}\n"
        print(menu)

if __name__ == "__main__":
    app = App()
    app.start()
