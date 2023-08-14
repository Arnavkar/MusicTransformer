
import os
def setup():
    print(os.getcwd())
    #if not os.exist("./content/gansynth"):
        #os.mkdir("./content/gansynth")

setup()
 

'''Install Magenta
print('Copying data from GCS...')
!rm -r /content/gansynth &>/dev/null
!mkdir /content/gansynth
!mkdir /content/gansynth/midi
!mkdir /content/gansynth/samples

# Get default MIDI (Bach Prelude)
!curl -o /content/gansynth/midi/bach.mid http://www.jsbach.net/midi/cs1-1pre.mid
MIDI_SONG_DEFAULT = '/content/gansynth/midi/bach.mid'
!curl -o /content/gansynth/midi/riff-default.mid http://storage.googleapis.com/magentadata/papers/gansynth/midi/arp.mid
MIDI_RIFF_DEFAULT = '/content/gansynth/midi/riff-default.mid'
'''