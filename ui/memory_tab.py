import streamlit as st
import logging
from typing import TYPE_CHECKING, Dict, List, Any

if TYPE_CHECKING:
    from lucid_memory.controller import LucidController
    from lucid_memory.memory_node import MemoryNode

def render_memory_tab(controller: 'LucidController', session_state: st.session_state):
    """Renders the Memory Node exploration tab."""
    # --- GET Nodes DICT *Inside Func Scope* ---
    nodes_dict: Dict[str, 'MemoryNode'] = controller.get_memory_nodes() # Needs to happen inside
    node_count = len(nodes_dict)
    st.header(f"🧠 Explore Memory Nodes ({node_count} Loaded)")


    # --- Controls (Filter, Sort, Refresh) ---
    col_filter, col_sort, col_refresh = st.columns([2, 2, 1])
    with col_filter:
        search_term = st.text_input("Filter nodes:", key="mem_node_filter", placeholder="Filter by ID, tag, summary...")
    with col_sort:
        sort_options = { # Define sort dictionary here
            "Seq Index": lambda item: (getattr(item[1], 'sequence_index', float('inf')), item[0]),
            "Node ID": lambda item: item[0],
            "Parent ID": lambda item: (getattr(item[1], 'parent_identifier', 'z'), item[0]) # Sort empty parents last
        }
        selected_sort_key = st.selectbox("Sort By:", options=list(sort_options.keys()), index=0, key="mem_node_sort")
    with col_refresh:
         st.caption(" ") # Align
         if st.button("🔄 Refresh", key="mem_refresh_btn"): st.rerun()

    # ==============================================
    # --- Logic Moved INSIDE the render function ---
    # ==============================================

    # --- Filtering ---
    items_to_display = nodes_dict.items() # START with all items
    if search_term:
        search_lower = search_term.lower()
        # Perform filtering operation -> Create NEW list based on filter.. OK..
        items_to_display = [
            (nid, node) for nid, node in nodes_dict.items()
            if search_lower in nid.lower() # Check Node ID..
            or search_lower in node.summary.lower() # Check Summary text..
            or any(search_lower in tag.lower() for tag in node.tags) # Check Tags..
            or any(search_lower in concept.lower() for concept in getattr(node, 'key_concepts', [])) # Check Concepts..
        ]
        # Feedback user Number FOUND helpful YES..
        st.caption(f"Displaying {len(items_to_display)} nodes matching filter.")


    # --- Sorting ---
    # Sort the (potentially filtered) items
    try:
      sorted_node_items = sorted(items_to_display, key=sort_options[selected_sort_key])
    except Exception as sort_err:
       sorted_node_items = list(items_to_display); # Fallback -> Unsorted.
       st.warning(f"Sort Error: {sort_err}", icon="⚠️")


    # --- Display Nodes Area ---
    with st.container(height=700): # Scroll Container
        if not sorted_node_items: # Use the final sorted items list.
             st.caption("(No memory nodes match criteria or graph is empty.)")
        else:
            for node_id, node in sorted_node_items: # Loop the sorted list.. ok..
                 # >>> Display Each node Detail in Expander >>> LOGIC Ok as BEfore...
                 with st.expander(f"Node: `{node_id}`", expanded=False):
                      # Build details string list
                      details_md = []
                      # ... (Append metadata, summary, tags, concepts, deps, outs, embedding status) ...
                      seq=getattr(node, 'sequence_index','?'); parent=getattr(node,'parent_identifier','?')
                      details_md.append(f"**Seq:** {seq} | **Parent:** `{parent}`")
                      details_md.append(f"**Summary:**\n```text\n{(node.summary or '(None)')}\n```")
                      tags = f"`{', '.join(node.tags)}`" if node.tags else "_None_"; details_md.append(f"**Tags:** {tags}");
                      kcs=getattr(node,'key_concepts',[]); details_md.append(f"**🔑 Concepts ({len(kcs)}):**"); details_md.extend([f"  - `{c}`" for c in kcs] if kcs else ["    _None_"]);
                      deps=getattr(node,'dependencies',[]);
                      if deps: details_md.append(f"**🔗 Deps ({len(deps)}):**"); details_md.extend([f"  - `{d}`" for d in deps]);
                      outs=getattr(node,'produced_outputs',[]);
                      if outs: details_md.append(f"**💡Outs ({len(outs)}):**"); details_md.extend([f"  - `{o}`" for o in outs]);
                      has_emb = getattr(node, 'embedding', None) is not None;
                      embDim = len(node.embedding) if has_emb else 0; embStatus =f"✅ {embDim}D" if has_emb else "❌"; details_md.append(f"**🧬 Embedding:** {embStatus}")
                      # --> Display aggregated DETAILS String -->
                      st.markdown("\n\n".join(details_md), unsafe_allow_html=False)

                      # --> Add optional Raw JSON view Toggle -->
                      if st.toggle("View Raw Data", key=f"raw_json_toggle_{node_id}", value=False):
                            try:
                               node_data = node.to_dict(); node_data.pop('embedding', None); # Hide vector.. OK..
                               st.json(node_data, expanded=False)
                            except Exception as json_e: st.error(f"JSON View Error: {json_e}")

      # >>>>>==== End OF the Node Display LOOP ==== <<<<<