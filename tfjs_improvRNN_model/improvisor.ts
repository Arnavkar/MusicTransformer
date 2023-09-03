//REMEMBER TO activate py27 environment

//const Tone = require('tone')
//require('@tensorflow/tfjs-node');
//const _ = require('lodash');
const core = require('@magenta/music/node/core');
const mm = require('@magenta/music/node/music_rnn');
const Tonal = require('tonal')
const Detect = require("tonal-detect")
const maxApi = require("max-api");


import { zeros } from "@tensorflow/tfjs-node";
import {TimeSettings, ChordProg, NumBeats, BeatValue, BeatDivision, MidiNote} from "./utils"

let improvisor: Improvisor;

async function loadModel(path:string): Promise<any> {
    // Using the Improv RNN pretrained model from https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn
    let rnn = new mm.MusicRNN(path);
    return await rnn.initialize();
}

class Improvisor {
    model: any;
    timeSettings: TimeSettings;
    currentChordProg: ChordProg;

    constructor(timeSettings: TimeSettings) {
        this.model = null; //loadModel('https://storage.googleapis.com/download.magenta.tensorflow.org/tfjs_checkpoints/music_rnn/chord_pitches_improv');
        this.timeSettings = timeSettings;
        this.currentChordProg = [];

        this.model.initialize();
    }

    updateTimeSettings(timeSettings: TimeSettings) {
        this.timeSettings = timeSettings;
    }

    updateChordProg(chordProg: ChordProg) {
        this.currentChordProg = chordProg;
    }
}

//retrieve time Settings from Max Patch, initialize Improvisor
maxApi.addHandler("getTimeSettings", (numbeats:NumBeats, beatvalue:BeatValue, measuredbeat:BeatDivision, isDotted:boolean, bpm:number) => {
    const timeSettings:TimeSettings = {
        numbeats: numbeats,
        beatvalue: beatvalue,
        measuredbeat: measuredbeat,
        isDotted: isDotted,
        bpm: bpm
    }
    improvisor = new Improvisor(timeSettings);
});


maxApi.addHandler("getNotes", (...midiNotes) => { //need to use spread operator to grab all values
    let notes = [];
    let midiNotesArray = [];
    for (let i = 0; i < midiNotes.length; i+=4) {
        if (i == 0) {
            let midinote : MidiNote = {
                pitch: midiNotes[i],
                velocity: midiNotes[i+1],
                duration: midiNotes[i+2]/1000,
                deltaTime: 0 //start
            }

            midiNotesArray.push(midinote);
        } else {
        }
        //duration received in ms, need to convert to seconds
        notes.push({
                    pitch: midiNotes[i],
                    startTime: i * 0.5
                });
    }

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










