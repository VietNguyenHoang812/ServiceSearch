from src.infra import neo4j_driver as driver


def get_node(node_id: str):
    query = """
        MATCH (n{id: $node_id})
        RETURN n
    """

    records, summary, keys = driver.execute_query(
        query,
        node_id=node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_father_node(node_id: str):
    query = """
        MATCH (c_o{id: $node_id})<-[r:HAS]-(n)
        RETURN n
    """

    records, summary, keys = driver.execute_query(
        query,
        node_id=node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_children_nodes(node_id: str):
    # Node Content
    if "C" in node_id:
        return []
    # Note Heading
    elif "H" in node_id:
        query = """
            MATCH (c_o:Heading {id: $node_id})-[r:HAS]->(n)
            RETURN n
        """
    # Note Document
    else:
        query = """
            MATCH (c_o:Document {id: $node_id})-[r:HAS]->(n:Heading)
            RETURN c_o, n
        """
        
    records, summary, keys = driver.execute_query(
        query,
        node_id=node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_sibling_nodes(node_id: str):
    # Node Content
    if "C" in node_id:
        query = """
            MATCH (c_o:Content {id: $node_id})<-[r:HAS]-(h:Heading)
            MATCH (h)-[r:HAS]->(n:Content)
            RETURN n
        """
    # Note Heading
    elif "H" in node_id:
        query = """
            MATCH (c_o:Heading {id: $node_id})-[r:IS_NEXT_TO]-(n:Heading)
            RETURN c_o, n
        """
    else:
        query = """
            MATCH (c_o:Document {id: $node_id})<-[]-(n)
            RETURN n
        """

    records, summary, keys = driver.execute_query(
        query,
        node_id=node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_metadata_node(node_id: str):
    id_doc, id_file_type = node_id.split("_")[0], node_id.split("_")[1]
    if id_file_type.startswith("VB") and len(id_file_type) > 2:
        meta_data_node_id = f"{id_doc}_{id_file_type}_C0"
    else:
        meta_data_node_id = f"{id_doc}_VB_C0"
    
    query = """
        MATCH (n{id: $node_id})
        RETURN n
    """

    records, summary, keys = driver.execute_query(
        query,
        node_id=meta_data_node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_doc_node(metadata_node_id: str):    
    query = """
        MATCH (c_o{id: $node_id})<-[r:HAS]-(n:Document)
        RETURN n
    """

    records, summary, keys = driver.execute_query(
        query,
        node_id=metadata_node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list