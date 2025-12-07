
from Core.Chunk.SchemaChunk import SchemaChunk
import itertools
class ChunkFactory:
    def __init__(self): self.c=itertools.count()
    def schema_to_chunks(self,sch):
        chunks=[]
        tnames=sch['table_names']
        colnames=sch['column_names']
        ctypes=sch['column_types']
        pks=set(sch['primary_keys'])
        tmap={i:t for i,t in enumerate(tnames)}
        for t in tnames:
            chunks.append(SchemaChunk(f"c{next(self.c)}","table",table=t))
        for idx,(ti,cn) in enumerate(colnames):
            chunks.append(SchemaChunk(f"c{next(self.c)}","column",table=tmap[ti],column=cn,datatype=ctypes[idx],pk=idx in pks))
        return chunks
