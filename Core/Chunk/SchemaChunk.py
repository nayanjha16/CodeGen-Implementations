
class SchemaChunk:
    def __init__(self,id,kind,table=None,column=None,datatype=None,pk=False,fk=None):
        self.chunk_id=id; self.kind=kind; self.table=table; self.column=column
        self.datatype=datatype; self.pk=pk; self.fk=fk or []
        self.text=str(self)
    def __str__(self):
        if self.kind=='table': return f"Table {self.table}"
        return f"Column {self.table}.{self.column} {self.datatype}"
