1) SETUP python packages:
    
    ***pip install -r SECONDARY/requirements.txt***

2) SETUP Jupyterlab:

    ***. venv/bin/activate && pip install jupyter lab***

3) START Jupyter lab:
    ***. venv/bin/activate && nohup jupyter lab &***

    ***. venv/bin/activate &&  jupyter notebook list***

    ***COPY LINK OPEN IN BROWSER***

4) RUN backtesting excutor in console: 

    ***. venv/bin/activate && pip install . && python -m SRC.BACKTESTING.backtesting_runner***

5) RUN backtesting presenter in Jupyter notebook: 
    
    ***Open and Play all of backtesting_presenter.ipynb***
