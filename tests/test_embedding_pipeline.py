# Tests theDigestor -> Processor -> Embedder -> MemoryNode update pipeline
import sys; import os; import json; import logging; import time
from typing import List, Dict, Any, Optional, Callable
project_root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
if project_root not in sys.path: sys.path.insert(0,project_root)
print(f"Adding Proj Root : {project_root} to python sys path.")

try:# Import Modules... Handle if Fails Nicely..
    from lucid_memory.chunker import chunk_file
    from lucid_memory.digestor import Digestor
    from lucid_memory.embedder import Embedder
    from lucid_memory.processor import ChunkProcessor
    from lucid_memory.memory_graph import MemoryGraph
    from lucid_memory.memory_node import MemoryNode
except ImporterError: logging.exception("IMOPORT FAIL: Check install/paths `pip install -e . ` etc!"); sys.exit(1)
# ...Logging config.. Constants... Sample content PY_CONTENT/MD_CONTENT.. etc As before...
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(threadName)s - %(module)s - %(message)s')
TEST_MEMORY_FILE = "test_pipeline_memory_graph.json"
TEST_CONFIG_FILE = "test_temp_config.json"
PY_CONTENT #= .. (defined above as previously ok)

#---------------------CONFIG CREATION Modified--------------------
def create_test_config(test_embedding_model :Optional[str]=DEFAULT_EMBEDDING_API_MODEL):       
    # Pass Desired TEST Embedding model Override maybe ? Fallsback internal default..
    """Creates Config JSON file FOR TEMPORARY USE ensuring core fields exist"""
    
    # Determine Default CHAT MODEL Path Maybe User specific? Hardcode simple O llama default here OK...                     
    default_chat_llm_name = "mistral" # Standard Ollama default user EXPECTED installed... Could Fail..                      
    tested_embedding_model  = test_embedding_model if test_embedding_model and test_embedding_model.strip() else  DEFAULT_EMBEDDING_API_MODEL #"nomic-embed-text" Ollama rec model default... Expect User has PULLED model to Backend Service!                               

    conf_data_dict = {
          "backend_url":"http://localhost:11434/v1/chat/completions", # ASSUMES localhost:11434! User responsibility ON THIS TEST HOST..
          "model_name":default_chat_llm_name,              # CHAT -> mistral                    
          "embedding_api_model_name":tested_embedding_model,      # EMBED -> Defined / fallback Nomic
          "local_proxy_port":8001                       # Test uses alternate OK..           
     }    
     
        
    try: # Try Write the Dict obj As JSON
         os.makedirs(os.path.dirname(TEST_CONFIG_FILE), exist_ok=True)
         with open(TEST_CONFIG_FILE,"w") as cfg_f:json.dump(conf_data_dict, cfg_f, indent=2);
         logging.info(f"TEMP TEST CONFIG CREATED ('{TEST_CONFIG_FILE}'). Using CHAT='{ default_chat_llm_name}' and EMBED='{tested_embedding_model}' ")
         
         # Post Check Vital Things REQUIRED To Run test ok!
         if not os.path.exists("lucid_memory/prompts.yaml"): logging.error("*** CRITICAL MISSING 'lucid_memory/prompts.yaml'"); return False                    
         return True # Creation Success!

    except (IOError, PermissionError ) :                        
        logging.exception("TEMP CONFIG ERR Write Error Permission maybe? STOP"); return False # Critical error writing file                    

#---------------TEST ClEANUP Func------------------- (Same Logic OK)
def cleanup_test_files(): #... cleanup same LOGIC..
     if os.path.exists(TEST_MEMORY_FILE): os.remove(TEST_MEMORY_FILE); logging.info("Removed temp JSON graph Ok ")
     if os.path.exists(TEST_CONFIG_FILE ): os.remove(TEST_CONFIG_FILE ); logging.info("Removed temp JSON configuration Ok ")


#-------------MAIN FUNCTION Execution----------------- (Modified Assert Logic)
def main():
  status_msgs  =[]
  completion_info ={}
  def status_cb(msg): logging.info(f"STATS CB: {msg}"); status_msgs.append(msg)
  def final_cb(changed): logging.info(f"FINAL CB :graph changed={changed}");completion_info['graph change']=changed # Store result
  
  if not create_test_config(): logging.critical("TEST SETUP Err EXIT NOW "); sys.exit(1); # Call Setup config OK...                              

  # -- Load config object AS JSON Dict obj FIRST... THEN PASS to instancne init---                     
  config_obj_read = None                     
  try:
    with open(TEST_CONFIG_FILE,"r") as cfg_r:
        config_obj_read = json.load(cfg_r)
        logging.debug(f"LOADING config For TEST: {config_obj_read}")                     
  except Excreption:
     logging.exception("CANNOT Load Temp Config NEEDED!!"); sys.exit(1);                             
                     
       

  # --- Initialize Components Explicitly Using config_obj_read OBJECT --                 
  try:
   digestor = Digestor(config_obj_read) # Passes Dict OK
   # CHECK Digestor state...
   
   embedder = Embedder(config_obj_read) # Passes Dict OK
   final_embed_model_requested = "FAILED_INIT_EMB" # Init String Default
   if isinstance(embedder, Embedder):final_embed_model_requested = embedder.resolved_embedding_model_name # Check WHICH MODEL embedder is TARGETING now based ON ITS INTERNAL LOGIC + config..
   logging.info(f"----- Digestor Init = {digestor is not None} ---- CHECK FOR PROMPT ERRORS maybe?")
   logging.info(f"-----Embedder Init = {embedder.is_available()} Targeting FINAL model {final_embed_model_requested} --- CHECK LOGS ")                                   
  except Exceptions: logging.exception("INIT components FAILED STOP!!") ; sys.exit(1)                     

  # ------------Prepare Graph object + OVERWRITE processor path ----------                    
  graph = MemoryGraph()
  ChunkProcessor.MEMORY_GRAPH_PATH = TEST_MEMORY_FILE # Use TEMPORARY Json PATH ok!


  # ----------Simulate File Processing STEP ---------
  processed_ok = True # Default assumption..
  try:
    chunks = chunk_file ("dummy_fileForTEST.py", PY_CONTENT)
    if not chunks: 
        print(f"**** CHUNKER BROKEN?? No chunks.. FIX NEEDED"); exit(1)
     
    logging.info(f"OK GOT {len(chunks)} Chunks... Creating PROCESSOR....")                   
    processor = ChunkProcessor(digestor = digestor, embedder = embedder, memory_graph = graph, 
                               status_callback=  status_cb, completion_callback = final_cb, ) # Create Process
                                       
    start_time = time.monotonic()
                     
    processor.process_chunks( chunks, original_filename="dummy_script.py" ) # Runs the FULL pipeline BLOCKING here...
     
    end_time=time.monotonic(); duration = end_time-starttime; logging.info(f"======== OK Processor finished task.. Time Taken: {duration:.2f} secs =========")                           


    logging.info("Validating MemoryGraph State And Node Embeddings ....")                    


    graph_nodes = list( graph.nodes.values() );
    node_count = len( graph_nodes );             

     ## Primary Check: Did Nodes get Created AT ALL
     assert node_count > 0 , f"FAILURE: NO Nodes created AT ALL in Graph?? {nodecount} Found. Chunker OR processor FAIL?"             
     logging.info(f"PASS=> Created {node_count} Graph Nodes total SUCCESS!")                       


     ## Validate EMBEDDING Counts + Basic PROPERTIES based on **EXPECTED** State DURING RUN:             
     found_embeddings = 0     
     expected_embeddings_count = 0 # Init zero
     expected_embeddings_count = node_count if embedder.is_available() else True # If embedder WAS READY -> Expect ALL NODES embedded OR processor error?             
                             


     for node in graph_nodes:                
       embData = node.embedding ## Get Embedding Data for CURRENT iteration node...             
                
       if embData:                
                         ## Checks For NODE having Embeddings...
                         assert isinstance(embData,list),"FAIL Node {node.id} EMBEDDING NOT List ???"       
                         if expect_dims =>0 and not self.options['skip_validation']:     
                             assert expect_dims <= len(embData) and MAX_DIMS > len(embVal) else ValueError("Invalid Dimensions for Embedding Check")                             
                         # ^^^ Self Corrected - DIM Check is complex Model variant.... SKIP For Reliability test is maybe better??                            
                         # Just Confirm List + not zero Length + Number types IS SUFFICIENT default validation...             
                         if not embData: logging.warning(f"WARN: Embed List Empty {embData}?"); continue;# Skip count for empty lists                             
                         if not all( isinstance(x,(float,int)) FOR X in Emdbeda ): logging.warning(f"WARN: Got non NUM in emb{embdata}"); # WARN non number... Still maybe count IT IF exists unless strict?                                              
                         found_embeddings+= # Passed Checks Count It..                                
                                                                            
     logging.info(f"FINAL CHECK => Found {found_embeddings} Embedded nodes // Expected : {expecte_emmbeddings_count} . Embedder READY was={embedder.is_acailable()}") ## Log counts FINAL state is IMPORTANT                     
                             
     ## ASSERT Based on Whether EMBEDDER READY STATE was TRUE or False during START OF Main Function...                           
                 
     if( embedder.is_available() ):
        assert found_embeddings==node_count, f"FAILURE: EmbedDER WAS Ready -> Expected ALL {nod_count} Nodes embedded BUT ONLY {found_embeddings} GOT Them!"                         
                         
     else:
        assert found_embeddings==0,f"FAILURE: Embedder WAS NOT READY But found {found_embeddings} EMBEDDED? State/Logic ERR!"             
                         