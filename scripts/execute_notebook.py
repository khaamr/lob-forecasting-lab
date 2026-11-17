#execute every cell and preserve errors for diagnosis
import json
import sys
import time
from pathlib import Path as path
import nbformat
from nbclient import NotebookClient as notebook_client
root=path(__file__).resolve().parents[1]
filename=root/'lob_reproduction_future_work.ipynb'
pass_number=int(sys.argv[1]) if len(sys.argv)>1 else 1
notebook=nbformat.read(filename,as_version=4)
start=time.perf_counter()
client=notebook_client(notebook,timeout=3600,kernel_name='lob-research',resources={'metadata':{'path':str(root)}})
def progress(cell,cell_index,**kwargs):
    if cell.cell_type=='code':
        print(f'executing cell {cell_index+1}/{len(notebook.cells)}',flush=True)
client.on_cell_start=progress
try:
    client.execute()
finally:
    nbformat.write(notebook,filename)
errors=[o for c in notebook.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
assert not errors
plots=sum('application/vnd.plotly.v1+json' in o.get('data',{}) for c in notebook.cells for o in c.get('outputs',[]))
assert plots>=27,plots
report=dict(pass_number=pass_number,seconds=time.perf_counter()-start,cells=len(notebook.cells),executed_code_cells=sum(c.cell_type=='code' for c in notebook.cells),plotly_outputs=plots,errors=len(errors))
(root/f'results/notebook_execution_{pass_number}.json').write_text(json.dumps(report,indent=2))
print(report,flush=True)
