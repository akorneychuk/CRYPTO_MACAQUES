1) SETUP python environment [python 3.10.x  | venv]:
    ```
    #INSTALL PYTHON
    cd ~/Downloads
    curl -O https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
    tar -xzf Python-3.10.13.tgz
    cd Python-3.10.13

    ./configure --prefix=$HOME/python310 --enable-optimizations

    make -j$(sysctl -n hw.ncpu)
    make install

    echo 'export PATH="$HOME/python310/bin:$PATH"' >> ~/.zshrc
    echo 'export PATH="$HOME/python310/bin:$PATH"' >> ~/.bash_profile
    source ~/.bash_profile && source ~/.zshrc

    mkdir -p ~/bin
    ln -s ~/python310/bin/python3.10 ~/bin/python3
    nano ~/.zshrc #>> export PATH="$HOME/bin:$PATH"
    nano ~/.bash_profile #>> export PATH="$HOME/bin:$PATH"
    source ~/.zshrc   # or source ~/.bash_profile
    source ~/.bash_profile   # or source ~/.bash_profile

    rm ~/bin/pip
    ln -s ~/python310/bin/pip3.10 ~/bin/pip3
    ```

2) SETUP python packages:
    
    ***. venv/bin/activate && pip install -r SECONDARY/requirements.txt***

3) SETUP Jupyterlab:

    ***. venv/bin/activate && pip install jupyter lab***

4) START Jupyter lab:

    ***. venv/bin/activate && nohup jupyter lab &***

    ***. venv/bin/activate &&  jupyter notebook list***

    ***COPY LINK OPEN IN BROWSER***

5) RUN backtesting excutor in console: 

    ***. venv/bin/activate && pip install . && python -m SRC.BACKTESTING.backtesting_runner***

6) RUN backtesting presenter in Jupyter notebook: 
    
    ***Open and Play all of backtesting_presenter.ipynb***
