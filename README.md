# ***CLIP-Vectorizer***
This is openAI's CLIP model based API that creates text and image vector-embeddings to be stored and query a vector database.

## ***Steps To run on localhost using Docker***
- Make sure Docker is installed and running (and using WSL2 engine if in windows).

- Follow the steps given in [Nvidia docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install nvidia drivers for your distribution (WSL for windows).

- If CUDA drivers are not present or GPU access is not provided to the container, then it will automatically default to computing on CPU.

- This command builds the image to be run inside a container  
  > `docker build -t vectorizer .`

- Run the program inside a container using
  > `docker run -it --gpus all vectorizer -p 5000:5000`  

## ***API routes***
- `/text_embedd` 
  > Post route for sending text to be embedded.   
  > Example Input JSON:  
  > {  
  > &emsp;"text" : "Your text here",  
  > &emsp;"normalized" : "True"  // Default = True  
  > }
- `/image_embedd`
  > Post route for sending images to be embedded.   
