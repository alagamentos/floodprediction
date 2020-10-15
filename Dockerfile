FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN pip install jupyter notebook

RUN mkdir /home/TCC
RUN mkdir /home/TCC/data
RUN mkdir /home/TCC/src

ADD requirements.txt /home/TCC

RUN pip install -r /home/TCC/requirements.txt

ADD data/ /home/TCC/data
ADD src/ /home/TCC/src

EXPOSE 8889

CMD ["jupyter", "notebook", "--port=8889", "--no-browser",\
      "--ip=0.0.0.0", "--allow-root"]
