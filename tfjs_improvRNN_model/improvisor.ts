//REMEMBER TO activate py27 environment

//const Tone = require('tone')
//require('@tensorflow/tfjs-node');
//const _ = require('lodash');
import * as core from "@magenta/music/node/core";
import { MusicRNN } from "@magenta/music/node/music_rnn";
//const core = require('@magenta/music/node/core');
const mm = require('@magenta/music/node/music_rnn');
const Tonal = require('tonal')
const Detect = require("tonal-detect")
const maxApi = require("max-api");

import { zeros } from "@tensorflow/tfjs-node";
import {TimeSettings, ChordProg, NumBeats, BeatValue, BeatDivision, Note} from "./utils"
import type { NoteSequence } from "@magenta/music/node/protobuf/index.d.ts";

let improvisor: Improvisor;

async function loadModel(path:string): Promise<any> {
    // Using the Improv RNN pretrained model from https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn
    let rnn = new MusicRNN(path);
    return await rnn
}

function quantizeNotes(notes: Note[], timeSettings: TimeSettings): Note[] {
    const quantizedNotes = notes.map((note) => {
        const quantizedNote = {
            pitch: note.pitch,
            velocity: note.velocity,
            duration: note.duration,
            startTime: note.startTime
        }
        return quantizedNote;
    });
    return quantizedNotes;
}

class Improvisor {
    model: any;
    timeSettings: TimeSettings;
    currentChordProg: ChordProg;
    inputNotes: Note[] | null;
    quantizedNotes: NoteSequence | null;

    constructor(timeSettings: TimeSettings) {
        this.model = null; //loadModel('https://storage.googleapis.com/download.magenta.tensorflow.org/tfjs_checkpoints/music_rnn/chord_pitches_improv');
        this.timeSettings = timeSettings;
        this.currentChordProg = [];
        this.inputNotes = null;
        this.quantizedNotes = null;
        //this.model.initialize();
    }

    updateTimeSettings(timeSettings: TimeSettings) {
        this.timeSettings = timeSettings;
    }

    updateChordProg(chordProg: ChordProg) {
        this.currentChordProg = chordProg;
    }

    quantizeNotes(){
        if (this.inputNotes){
            let notes = [];
            for (let i = 0; i < this.inputNotes.length; i++){
                notes.push({
                    pitch: this.inputNotes[i].pitch,
                    startTime: this.inputNotes[i].startTime,
                    endTime: this.inputNotes[i].startTime + this.inputNotes[i].duration,
                });
            }
            let quantizedSequence = core.sequences.quantizeNoteSequence(
                {
                    ticksPerQuarter: 480,
                    totalTime: this.timeSettings.bpm / 60 * this.timeSettings.numbeats,
                    quantizationInfo: {
                        stepsPerQuarter: 4
                    },
                    timeSignatures: [
                        {
                            time: 0,
                            numerator: this.timeSettings.numbeats,
                            denominator: this.timeSettings.beatvalue
                        }
                    ],
                    tempos: [
                        {
                            time: 0,
                            qpm: this.timeSettings.bpm
                        }
                    ],
                    notes
                },
                1
            );
            this.quantizedNotes = quantizedSequence;
        }
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
    if (improvisor){
        improvisor.updateTimeSettings(timeSettings);
    } else {
        improvisor = new Improvisor(timeSettings);
    }
    console.log(improvisor.timeSettings);
});
3

maxApi.addHandler("getNotes", (...midiNotes: number[]) => { //need to use spread operator to grab all values
    for (let i = 0; i < midiNotes.length; i+=4) {
        let note : Note = {
            pitch: midiNotes[i],
            velocity: midiNotes[i+1],
            duration: midiNotes[i+2]/1000,
            startTime: midiNotes[i+3] //already in seconds
        }
        if (improvisor.inputNotes === null) {
            improvisor.inputNotes = [];
        }
        improvisor.inputNotes.push(note);
    }
    console.log(improvisor.inputNotes);
    improvisor.quantizeNotes();
    console.log(improvisor.quantizedNotes);
    // console.log("MIDI NOTES ARRAY –––––––– \n")
    // console.log(midiNotesArray)
    // console.log("NOTES ARRAY –––––––– \n")
    // console.log(notes);
});




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










