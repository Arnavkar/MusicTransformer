
#Copy samples from coglab-5 to local machine
scp -r -v arnav@coglab-5:/home/arnav/MusicTransformerLink/PyModel/samples ./

#Copy samples from local machine to coglab-5
scp -r -v ./samples arnav@coglab-5:/home/arnav/MusicTransformerLink/PyModel/

#Copy models from coglab-5 to local machine
scp -r -v arnav@coglab-5:/home/arnav/MusicTransformerLink/PyModel/models ./

#Copy models from local machine to coglab-5
scp -r -v ./models arnav@coglab-5:/home/arnav/MusicTransformerLink/PyModel/





