import numpy as np

class Vector_store:
    def __init__(self) -> None:
        self.vector_data = {} # A dictionary to store vectors
        self.vector_index = {} # A dictionary for indexing structure for reterival
    
    def add_vector(self,vector_id,vector):
        """
        Add a vector to the vector store

        Arg:
            vector_id (str or int) : A unique id for the vector
            vector (nump.darray): the vector to be stored
        """
        self.vector_data[vector_id] = vector
        self.update_index(vector_id,vector)
    
    def get_vector(self,vector_id):
        """
        Get vector from vector store

        Args:
            vector_id (str or int) : A unique id for the vector

        Return : A nump.darray if not found return None

        """
        return self.vector_data.get(vector_id)

    def update_index(self,vector_id,vector):
        """
        Update the indexing of the vector

        Arg:
            vector_id (str or int) : A unique id for the vector
            vector (nump.darray): the vector to be stored

        """
        for exisiting_id,exisiting_vector in self.vector_data.items():
            similarity = np.dot(vector,exisiting_vector) / (np.linalg.norm(vector) * np.linalg.norm(exisiting_vector))
            if exisiting_id not in self.vector_index:
                self.vector_index[exisiting_id] = {}
            self.vector_index[exisiting_id][vector_id] = similarity

    def similar_vector(self,query_vector,num_results=5):
        """
        Find similarity vectors to query

        Args:
            query_vector (numpy.darray) : query vector
            num_results (int) : to get the num of matching results

        Returns:
            list : A list of tuples of form (vector_id,similarity)

        """
        results = []
        for vector_id,vector in self.vector_data.items():
            similarity = np.dot(query_vector,vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id,similarity))
        # Sort the similarity in descending order
        results.sort(key=lambda x:x[1] ,reverse=True)
        return results[:num_results]
