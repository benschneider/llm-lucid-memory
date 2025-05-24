import logging
import os
import sys

def main():
    """
    Runs the Lucid Memory Unified FastAPI Server.
    This is an entry point for console_scripts.
    """
    log_level_str = os.environ.get('LUCID_LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger("LucidServerRunner")

    try:
        import uvicorn
        from lucid_memory.server_api import app as fastapi_app, server_config
    except ImportError as e:
        logger.critical(f"Failed to import uvicorn or server_api components: {e}", exc_info=True)
        logger.critical("Ensure FastAPI, Uvicorn, and lucid_memory are installed correctly and paths are set up.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error during imports for server_runner: {e}", exc_info=True)
        sys.exit(1)

    # MODIFIED: Default host and port
    host = server_config.get("unified_server_host", "127.0.0.1")
    port = server_config.get("unified_server_port", 8081)
    workers = server_config.get("unified_server_workers", 1)
    reload_mode = server_config.get("unified_server_reload", False) # For dev

    host = os.environ.get('LUCID_SERVER_HOST', host)
    try:
        port = int(os.environ.get('LUCID_SERVER_PORT', str(port)))
        workers = int(os.environ.get('LUCID_SERVER_WORKERS', str(workers)))
    except ValueError:
        logger.warning(f"Invalid port or worker count in environment variables. Using defaults: host={host}, port={port}, workers={workers}")

    if os.environ.get('LUCID_SERVER_RELOAD', '').lower() in ['true', '1', 'yes']:
        reload_mode = True
        logger.info("Unified server reload mode enabled via environment variable (for development).")

    logger.info(f"Starting Lucid Memory Unified Server on http://{host}:{port} with {workers} worker(s). Reload: {reload_mode}")
    logger.info(f"Log level set to: {log_level_str}")

    app_string = "lucid_memory.server_api:app"

    try:
        uvicorn.run(
            app_string,
            host=host,
            port=port,
            workers=workers if not reload_mode else 1,
            reload=reload_mode,
            log_level=log_level_str.lower()
        )
    except RuntimeError as e:
        if "already an active HTTP server" in str(e).lower() or "error while attempting to bind on address" in str(e).lower() :
             logger.error(f"Failed to start Unified Server: Port {port} on {host} might already be in use.")
        else:
             logger.critical(f"Unified Server failed to start: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred while running the Unified Server: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Lucid Memory Unified Server has shut down.")

if __name__ == '__main__':
    main()