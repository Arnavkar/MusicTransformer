//REMEMBER TO activate py27 environment

//const Tone = require('tone')
//require('@tensorflow/tfjs-node');
const core = require('@magenta/music/node/core');
const mm = require('@magenta/music/node/music_rnn');
const Tonal = require('tonal')
const Detect = require("tonal-detect")
const maxApi = require("max-api");
const _ = require('lodash');

class MidiNote {
    constructor(pitch, velocity, time) {
        this.pitch = pitch;
        this.velocity = velocity;
        this.duration = time;
    }
}

class TimeSignature {
    constructor(numbeats,beatvalue) {
        this.numbeats = numbeats;
        this.beatvalue = beatvalue;
    }
}

async function loadModel(path) {
    // Using the Improv RNN pretrained model from https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn
    let rnn = new mm.MusicRNN(path);
    return await rnn.initialize();
}


//let model = loadModel('https://storage.googleapis.com/download.magenta.tensorflow.org/tfjs_checkpoints/music_rnn/chord_pitches_improv');

//need to use spread operator to grab all values
// maxApi.addHandler("getNotes", (...midiNotes) => {
//     let notes = [];
//     let midiNotesArray = [];
//     for (let i = 0; i < midiNotes.length; i+=3) {
//         //duration received in ms, need to convert to seconds
//         midiNotesArray.push(new MidiNote(midiNotes[i], midiNotes[i+1], midiNotes[i+2]/1000));
//         notes.push({
//                     pitch: midiNotes[i],
//                     startTime: i * 0.5
//                 });
//     }

//     _.forEach(midiNotesArray, (note) => {
//     // let notes = [];
//     // for (let i = 0; i < midiNotes.length; i++) {
//     //     if (seq[i] === -1 && notes.length) {
//     //     _.last(notes).endTime = i * 0.5;
//     //     } else if (seq[i] !== -2 && seq[i] !== -1) {
//     //     if (notes.length && !_.last(notes).endTime) {
//     //         _.last(notes).endTime = i * 0.5;
//     //     }
//     //     notes.push({
//     //         pitch: seq[i],
//     //         startTime: i * 0.5
//     //     });
//     //     }
//     // }
//     // if (notes.length && !_.last(notes).endTime) {
//     //     _.last(notes).endTime = seq.length * 0.5;
//     // }
//     // noteEventList = [];
//     // for (let i = 0; i < midiNotes.length; i+=2) {
//     //     const noteEvent = new Uint8Array([midiNotes[i], midiNotes[i+1]]);
//     //     noteEventList.push(noteEvent);
//     // }
// 	// const notes = midiNotes.map(Tonal.Note.fromMidi);
// 	// const possibleChords = Detect.chord(notes);
//     // console.log(noteEventList)
// });










