### CHAT WITH OPENEARTH


#### QUICK START


#### step 1 (Configs)


Setting  up example.env file. You can use ollama or openai apis.

**Ollama**
- Install ollama from https://ollama.com/ 
- Pull the required llm to be used
- Create langchain api keys  https://docs.smith.langchain.com/
- uncomment langchain langchain keys and fill the values in .env file.

**OpenAI**
- Uncomment openAI api variables and fill the values in .env file.

NOTE: Change file name from example.env to .env

Note: If Openai Apis Are Commented, Ollama Will Be Used Instead.

Set up the urls in config.yaml, add any urls, to be used by the bot.
    
      URLS:
	  	-"http://www.bonap.org/"
      	-"https://www.wildflower.org/plants-main"
    


#### step 2 (Runing the chat bot)
`source run.sh`

#### step 3 (Time to CHAT)
Start chatting with the bot and ask any question related to gardening. it will try to answer based on the urls added within the config file
