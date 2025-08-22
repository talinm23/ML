Semantic search system using RAG

This project is a semantic search system using Retrieval-Augmented Generation (RAG) to analyze and interact with PDF-formatted books. The goal is to enable efficient analysis, exploration, and conversation with the text data 

The process:

It takes the pdf of the book, splits it into sentences in the most basic way (using PYPDF2). The sentences get passed into the "thenlper/gte-large" model using the SentenceTransformer library. We get the vector embeddings back and save them in a MongoDB collection. Then using the same model above, we embed a query about the pdf (e.g. what is the author's name). It searches the MongoDB collection using a vector search pipeline based on the user's query and returns a list of matchig documents. Then the outputs along with the original query about the book are passed into Google's Gemma model (google/gemma-2b-it) and it responds with the answers. 

This project was inspired by this post: https://www.mongodb.com/developer/products/atlas/gemma-mongodb-huggingface-rag/


'''
Example output for the question:

    query = "What is the name of the author? what are the natural remedies? What are the most effective cures? list the natural remedies."

Output:

    <bos>Query: What is the name of the author? what are the natural remedies according to the book? list the natural remedies from the book.
    Continue to answer the query by using the Search Results:
    result:  Read what the legendary natural healer Dr
    result:  After  being  an award -winning  author,  after  writing  22 best -selling  books  many  in 9 languages or  more  and seeing  why they  are so successful,  I decided  to write  an educational  book  about  health  and self-help  for all those  people  who don’t have as much knowledge about medicine or natural healing as a fully educated practitioner
    result: My books have the purpose to educate people so that they can protect themselves and cure themselves with natural means
    result: This book is written for the person with  common  sense  and the will to achieve  optimum  health  without getting killed by the medical profession, pharmaceutical industry  or the hocus pocus new age “wannabe healers” an d the new crowd of naturopathic  practitioners  with  no formal  education,  experience  or real life knowledge about health
    result: He read books about natural nutrition a nd behavior regimens based on natural methods
    result: This book is a testament to God’s truth and healing power, and my loving gift to  you!  Be prepared, because once this second book hits the mass market they will attack me personally and will try to destroy my name and reputation, but they cannot attack or disprove my message and my  cure —God’s  cure  and Nature’s  cure
    result: In the pages to come, I will show you how to activate your body’s natural healing powers
    result:  Many  of my relatives,  friends,  and acqua intances have been cured of all kinds of health problems through my  method of  natural  healing
    result:         237 | P a g e   Chapter 10   HOW I CURED MY PATIENTS FROM  ALL KINDS OF HEALTH CHALLENGES   The Book of Health™      From my books, here are some summar ies I want to share with you
    result:  Nature  has cures  for everything  and without  any negative  side effects
    result: Source: Chinese  Medicine
    result: I studied natural healing, explored all available information, researched, learned and  discovered!  86 | P a g e   I discovered something “new” that is actually as old as humanity itself: The only help is self -help, and the only way to healing is self -healing
    result: Otherwise, they use helpful and harmless natural  medicines
    result:       37 | P a g e   Chapter 2   INSTINCT BASED MEDICINE®  The guide for healthy action    Nearly ever y health and self -healing book I have ever read is written in a manner that is too complicated, too specific, too hard to understand, too manipulative, too out of this world, too  new age  or too much  like traditional  medicine
    result: Batmanghelidj, the author of The Body’s Many Cries for Water, has seen  similar  results  of 136 | P a g e   the body’s  self-healing  powers  just by water  application
    result: They are riding the wave of natural cures, alternative health etc ., etc., etc
    result: These are the people  that ridicule natural healing methods as  quackery
    result:  “The only way to perfect health and for prevention of health challenges  is to be yourself  the way you were  meant  to be by nature, God  or the Universe  or whatever  you believe  in, it is the same  thing anyway) there is no healing force outside the human body.”  I am writing  this book  because  the medical  profession  murdered my grandmother 
    result: Look at how PERFECT nature is - why is it tha t for every illness, there is a plant, usually native to the local area where illness occurs, that is often the cure
    result: Review articles about diseases will either omit information about natural therapies or the information about natural treatments will be presented in such a biased or  negative way that no physician would want to use  it
    result: 1 | P a g e          2 | P a g e   THE  ONLY PATIENT CANCER CURE      Dr LEONARD COLDWELL3 | P a g e   IBMS® Instinct Based Medicine System®   The Only answer to® THE ONLY CANCER PATIENT CURE  Dr LEONARD COLDWELL   2nd Edition: Copyright © 2019, Cancer Patient  Advocate Foundation, a non -profit organi zation
    result:  I’ve spent  the past  45 years  showing  people  how  to activate  their body’s natural healing power
    result: The problem is, people like Bob give the ENTIRE natural cures industry a bad name
    result: Besi des stress relief, I used  only natural products to cure my patients, mainly natural cleanses  (cleanses  that  are  based  on  certified  organic  whole  foods)
    result:   They Think They Can Cure Cancer   I am so tired of people announcing that they have invented a natural cure for cancer then use this information to publish a book
    result: Source: Widely accepted by both western and alternative medical  communities
    result:  15 | P a g e   I know it will get even worse with the publication of this book because in it I tell you how easy it can be to cure canc er patients and how I did it
    result:  The only things that help the body to heal itself are natural elements like a healthy diet with plenty of water and fresh juices, correct  breathing  and exercise  and nutritional  supplements  with  no side effects
    result: From start to finish, I will show you how you can  rid yourself of disease using nothing else than the natural function of the body
    result: I learned from the Nati ve Indians in Canada and USA and  in South  America,  the natural  ways  of healing  and health
    result:  With  this motivation,  I read  every  book I could  find about  healing, orthodox medicine, natural healing, metaphysics and related matters, with the hope that I would find a way to heal my mother
    result: While studying naturopathic medicine, I discovered that new age stuff does not work
    result: Most of the time, these people have overcome life -threatening illnesses that they have cured through natural therapies like self -help and preventative techniques
    result:  There are many good reasons why physicians have 442 | P a g e   not started to use natural therapies:   • Physicians receive no education in medical school about the merits of natural treatments
    result: Who can I trust?”  “It will get better on its own.” “It’s just bad luck.”  “It’s genetic.”  “The doctors will cure me or the medication will help.”  This book is written for the person searching for health an d happiness who is willing to do his  own part to accelerate health and happiness and who is willing to take responsibility for his own well -being
    result:  • Only natural treatments are legal (the except ions are trauma care
    result:  • Turmeric  • Various  mushrooms  • Oleander  soup  • Coffee  Enemas  • Oxygen or Ozone therapy ( has some dangers Oxygen is safer  ) • Vitamin  D3, 150,000 to 200,000 IU per day or lots of  140 | P a g e   sunlight  • Raw food or macrobiotic  diets  • Full body and organ  cleanses  • DMSO and Cesium chloride  therapy  • Chinese Happy  Tree  • Honey with  cinnamon  • Organic Maple Syrup with baking soda, ½ tbsp Spoon baking soda on 2 tb sp Maple Syrup caramelized in a frying pan...eaten over the day
    result:   Belief #9 : “There is no known remedy for my illness 
    result: Look at how PERFECT nature is - why is it that for every illness, there is a plant, usually native to the local area where illness occurs, that is o ften the cure
    result:  I studied  people  who  produced  results  unknown  in traditional medicine
    result: There is no way to combine traditional medicine and natural medicine
    result:   Pine Needle Tea: Fortify Yourself with this Unusual Cancer -Killer and All -Around Health Tonic   Pine needle  tea has been used for centuries by Native Americans
    result:   Becoming a Naturopathic Practitioner   I finished my studies as  a Natural Healing Physician (ND)
    result: Hay, Heal Your Body, Heal Your Life & Dr
    result: That is because traditional medicine harms or even kills and natural medicine helps the body to heal itself
    result: My education taught me that healing is based on the basic  laws of nature
    result: However,  if you’re  looking  for a magic  pill or a miracle  cure that  will fix all of your  problems  without  you taking  charge  and control over your own life and health, then this book 187 | P a g e   is not for  you
    result:  Cancer, diabetes, heart related problems, arthritis and so much more can be cured with all natural extremely cheap treatments that have been proven safe and effec tive in many cases even for thousands of years
    result:  Don’t wait for a second longer to educate yourself about your body’s natural healing power, and start using it immediately! Making  educated  decisions  is the best  defense  against  illness  and the best way to return to optimum  health
    result: In addition to the natural diet component, Dr
    . Leonard Coldwell's book also recommends a variety of natural remedies, including:

    **Natural Remedies from the Book**

    1. Turmeric
    2. Various mushrooms
    3. Oleander soup
    4. Coffee enemas
    5. Oxygen or ozone therapy
    6. Vitamin D3
    7. Raw food or macrobiotic diets
    8. Full body and organ cleanses
    9. DMSO and Cesium chloride therapy
    10. Chinese Happy Tree
    11. Honey with cinnamon
    12. Organic Maple Syrup with baking soda, ½ tbsp Spoon baking soda on 2 tb sp Maple Syrup caramelized in a frying pan...eaten over the day
    13. Pine Needle Tea
    14. Hay, Heal Your Body, Heal Your Life & Dr<eos>


'''

There are two notebooks to run: step1 and step2. 

You must have a MongoDB and HuggingFace accounts and put the tokens in the secret keys in your Colab notebook.
It could be possible to run these notebooks to run locally too with your own Jupyter Notebooks.
For these codes, we used the T4 GPU to run it much faster than using the CPU (I believe it wasn't even running with the CPU). 

Step1:  
We first mount Google Drive and point to a PDF in Drive. Install and import PyPDF2 if needed.
Then we extract text from sentences: the code reads every page of the PDF with PyPDF2.PdfReader, concatenates text, then does a simple split on '. ' to get sentences. Removes newline characters in each sentence.
For the embeddings, the code loads the SentenceTransformers model thenlper/gte-large and defines get_embedding(text) to return a float vector for each sentence (skips empty strings).

Then we build the dataset. The code applies get_embedding to all sentences, producing a Series of vectors; converts to a DataFrame with:
- embedding_: the embedding vector for each sentence
- sentences: the original sentence text (without newlines)
  
Then we save the DataFrame to Google Drive as csv_saved/dataset_embedded.csv. (literal_eval was imported but not used)

Step2:





    
