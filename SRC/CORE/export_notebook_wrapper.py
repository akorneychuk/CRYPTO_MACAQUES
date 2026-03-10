from SRC.CORE._FUNCTIONS import PRODUCE_OUT_FOLDER_PATH

def export_notebook_wrapper(suffix=None):
	import sys

	sys.path.append('../../')
	sys.path.insert(1, '/CORE')
	sys.path.insert(1, '/CORE/debug_utils')
	from ipylab import JupyterFrontEnd
	from SRC.CORE.debug_utils import export_notebook

	app = JupyterFrontEnd()
	app.commands.execute('docmanager:save')
	export_notebook(PRODUCE_OUT_FOLDER_PATH(), suffix)