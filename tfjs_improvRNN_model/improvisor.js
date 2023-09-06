"use strict";
//REMEMBER TO activate py27 environment
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
//const Tone = require('tone')
//require('@tensorflow/tfjs-node');
//const _ = require('lodash');
const core = __importStar(require("@magenta/music/node/core"));
const music_rnn_1 = require("@magenta/music/node/music_rnn");
//const core = require('@magenta/music/node/core');
const mm = require('@magenta/music/node/music_rnn');
const Tonal = require('tonal');
const Detect = require("tonal-detect");
const maxApi = require("max-api");
let improvisor;
function loadModel(path) {
    return __awaiter(this, void 0, void 0, function* () {
        // Using the Improv RNN pretrained model from https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn
        let rnn = new music_rnn_1.MusicRNN(path);
        return yield rnn;
    });
}
function quantizeNotes(notes, timeSettings) {
    const quantizedNotes = notes.map((note) => {
        const quantizedNote = {
            pitch: note.pitch,
            velocity: note.velocity,
            duration: note.duration,
            startTime: note.startTime
        };
        return quantizedNote;
    });
    return quantizedNotes;
}
class Improvisor {
    constructor(timeSettings) {
        this.model = null; //loadModel('https://storage.googleapis.com/download.magenta.tensorflow.org/tfjs_checkpoints/music_rnn/chord_pitches_improv');
        this.timeSettings = timeSettings;
        this.currentChordProg = [];
        this.inputNotes = null;
        this.quantizedNotes = null;
        //this.model.initialize();
    }
    updateTimeSettings(timeSettings) {
        this.timeSettings = timeSettings;
    }
    updateChordProg(chordProg) {
        this.currentChordProg = chordProg;
    }
    quantizeNotes() {
        if (this.inputNotes) {
            let notes = [];
            for (let i = 0; i < this.inputNotes.length; i++) {
                notes.push({
                    pitch: this.inputNotes[i].pitch,
                    startTime: this.inputNotes[i].startTime,
                    endTime: this.inputNotes[i].startTime + this.inputNotes[i].duration,
                });
            }
            let quantizedSequence = core.sequences.quantizeNoteSequence({
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
            }, 1);
            this.quantizedNotes = quantizedSequence;
        }
    }
}
//retrieve time Settings from Max Patch, initialize Improvisor
maxApi.addHandler("getTimeSettings", (numbeats, beatvalue, measuredbeat, isDotted, bpm) => {
    const timeSettings = {
        numbeats: numbeats,
        beatvalue: beatvalue,
        measuredbeat: measuredbeat,
        isDotted: isDotted,
        bpm: bpm
    };
    if (improvisor) {
        improvisor.updateTimeSettings(timeSettings);
    }
    else {
        improvisor = new Improvisor(timeSettings);
    }
    console.log(improvisor.timeSettings);
});
3;
maxApi.addHandler("getNotes", (...midiNotes) => {
    for (let i = 0; i < midiNotes.length; i += 4) {
        let note = {
            pitch: midiNotes[i],
            velocity: midiNotes[i + 1],
            duration: midiNotes[i + 2] / 1000,
            startTime: midiNotes[i + 3] //already in seconds
        };
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
