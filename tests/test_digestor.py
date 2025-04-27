from lucid_memory.digestor import Digestor

def test_digestor_basic():
    raw_text = """def start_server(port):
        # Open socket
        # Bind port
        # Accept HTTP requests
    """
    digestor = Digestor()
    node = digestor.digest(raw_text, node_id="func_start_server")

    assert node.id == "func_start_server"
    assert "start_server" in node.raw
    assert isinstance(node.summary, str)
    assert isinstance(node.reasoning_paths, list)
    assert len(node.reasoning_paths) >= 1
    assert isinstance(node.tags, list)

def test_digestor_tagging_server():
    digestor = Digestor()
    node = digestor.digest("A function that runs a network server.", node_id="server_fn")
    assert "server" in node.tags
    assert "network" in node.tags