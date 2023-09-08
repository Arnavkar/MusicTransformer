const SAMPLE_LIBRARY = {
  'Vibraphone': [
      { note: 'A',  octave: 3, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.A3.wav' },
      { note: 'A',  octave: 4, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.A4.wav' },
      { note: 'A',  octave: 5, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.A5.wav' },
      { note: 'B',  octave: 3, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.B3.wav' },
      { note: 'B',  octave: 4, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.B4.wav' },
      { note: 'B',  octave: 5, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.B5.wav' },
      { note: 'C',  octave: 3, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.C3.wav' },
      { note: 'C',  octave: 4, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.C4.wav' },
      { note: 'C',  octave: 5, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.C5.wav' },
      { note: 'C',  octave: 6, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.C6.wav' },
      { note: 'D',  octave: 3, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.D3.wav' },
      { note: 'D',  octave: 4, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.D4.wav' },
      { note: 'D',  octave: 5, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.D5.wav' },
      { note: 'D',  octave: 6, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.D6.wav' },
      { note: 'E',  octave: 3, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.E3.wav' },
      { note: 'E',  octave: 4, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.E4.wav' },
      { note: 'E',  octave: 5, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.E5.wav' },
      { note: 'E',  octave: 6, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.E6.wav' },
      { note: 'F',  octave: 3, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.F3.wav' },
      { note: 'F',  octave: 4, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.F4.wav' },
      { note: 'F',  octave: 5, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.F5.wav' },
      { note: 'F',  octave: 6, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.F6.wav' },
      { note: 'G',  octave: 3, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.G3.wav' },
      { note: 'G',  octave: 4, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.G4.wav' },
      { note: 'G',  octave: 5, file: 'Samples/Vibraphone/Vibraphone.sustain.ff.G5.wav' },
  ]
};

const OCTAVE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

let audioContext = new AudioContext();

async function fetchSample(path) {
  const response = await fetch(encodeURIComponent(path));
  const arrayBuffer = await response.arrayBuffer();
  return await audioContext.decodeAudioData(arrayBuffer);
}

function noteValue(note, octave) {
    return octave * 12 + OCTAVE.indexOf(note);
}

function flatToSharp(note) {
  switch (note) {
    case 'Bb': return 'A#';
    case 'Db': return 'C#';
    case 'Eb': return 'D#';
    case 'Gb': return 'F#';
    case 'Ab': return 'G#';
    default:   return note;
  }
}

function getNoteDistance(note1, octave1, note2, octave2) {
    return noteValue(note1, octave1) - noteValue(note2, octave2);
}

async function getSample(instrument, noteAndOctave){
  //(\w[b#]?) captures single word character followed by a b or # , (\d) captures digit
  let [, requestedNote, requestedOctave] = /^(\w[b#]?)(\d)$/.exec(noteAndOctave);
  requestedOctave = parseInt(requestedOctave, 10);
  requestedNote = flatToSharp(requestedNote);

  let sampleBank = SAMPLE_LIBRARY[instrument];
  let nearestSample = getNearestSample(sampleBank, requestedNote, requestedOctave);
  let distance = getNoteDistance(requestedNote, requestedOctave, nearestSample.note, nearestSample.octave);
  
  const audioBuffer = await fetchSample(nearestSample.file);
  return ({
    audioBuffer: audioBuffer,
    distance: distance
  });
}

function getNearestSample(sampleBank, note, octave) {
    let sortedBank = sampleBank.slice().sort((sampleA, sampleB) => {
      let distanceToA =
        Math.abs(getNoteDistance(note, octave, sampleA.note, sampleA.octave));
      let distanceToB =
        Math.abs(getNoteDistance(note, octave, sampleB.note, sampleB.octave));
      return distanceToA - distanceToB;
    });
    return sortedBank[0];
}

async function playSample(instrument, note,destination, delaySeconds = 0){
    let gainNode = audioContext.createGain()
    gainNode.gain.value = 0.02;
    gainNode.connect(destination);

    let {audioBuffer, distance} = await getSample(instrument, note);
    let playbackRate = Math.pow(2, distance / 12);
    let bufferSource = audioContext.createBufferSource();

    bufferSource.buffer = audioBuffer;
    bufferSource.playbackRate.value = playbackRate;

    bufferSource.connect(gainNode);
    bufferSource.start(audioContext.currentTime + delaySeconds);
    console.log(`${note} was played`)
}

function startLoop(instrument,note,destination,loopLengthSeconds, delaySeconds) {
  playSample(instrument, note,destination,delaySeconds);
  setInterval(() => playSample(instrument, note, destination, delaySeconds), loopLengthSeconds * 1000)
};

//Launch
async function launch(withReverb = false, pathToReverb = 'AirwindowsImpulses/AirportTerminal.wav'){
  if (!withReverb) {
    startLoop('Vibraphone','F4',  audioContext.destination, 19.7, 4.0);
    startLoop('Vibraphone','Ab4', audioContext.destination, 17.8, 8.1);
    startLoop('Vibraphone','C5',  audioContext.destination,21.3, 5.6);
    startLoop('Vibraphone','Db5', audioContext.destination,22.1, 12.6);
    startLoop('Vibraphone','Eb5', audioContext.destination,18.4, 9.2);
    startLoop('Vibraphone','F5',  audioContext.destination,20.0, 14.1);
    startLoop('Vibraphone','Ab5', audioContext.destination,17.7, 3.1);
  } else {
    const audioBuffer = await fetchSample(pathToReverb)
    let convolver = audioContext.createConvolver();
    convolver.buffer = audioBuffer;
    convolver.connect(audioContext.destination);
    startLoop('Vibraphone','F4',  convolver, 19.7, 4.0);
    startLoop('Vibraphone','Ab4', convolver, 17.8, 8.1);
    startLoop('Vibraphone','C5',  convolver,21.3, 5.6);
    startLoop('Vibraphone','Db5', convolver,22.1, 12.6);
    startLoop('Vibraphone','Eb5', convolver,18.4, 9.2);
    startLoop('Vibraphone','F5',  convolver,20.0, 14.1);
    startLoop('Vibraphone','Ab5', convolver,17.7, 3.1);
  }
}
audioContext.resume()
launch(true,'AirwindowsImpulses/PlateLarge.wav')
