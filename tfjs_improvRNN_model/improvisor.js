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
const core = __importStar(require("@magenta/music/node/core"));
const music_rnn_1 = require("@magenta/music/node/music_rnn");
const _ = require('lodash');
const Detect = require('tonal-detect');
const Tonal = require('tonal');
const maxApi = require("max-api");
let improvisor;
class Improvisor {
    constructor(timeSettings) {
        this.model = this.loadModel('https://storage.googleapis.com/download.magenta.tensorflow.org/tfjs_checkpoints/music_rnn/chord_pitches_improv');
        this.timeSettings = timeSettings;
        this.currentChordProg = [];
        this.inputNotes = [];
        this.quantizedNotes = core.sequences.createQuantizedNoteSequence();
    }
    loadModel(path) {
        // Using the Improv RNN pretrained model from https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn
        let rnn = new music_rnn_1.MusicRNN(path);
        rnn.initialize();
        return rnn;
    }
    updateTimeSettings(timeSettings) {
        this.timeSettings = timeSettings;
    }
    updateChordProg(chordProg) {
        this.currentChordProg = chordProg;
    }
    quantizeInputNotes() {
        // if (!this.model){
        //     throw new Error("Model not loaded!");
        // }
        var _a, _b, _c;
        // if (!this.model.isInitialized){
        //     throw new Error("Model not Initialized!");
        // }
        // if (!this.timeSettings){
        //     throw new Error("Time Settings not initialized!");
        // }
        if (this.inputNotes) {
            let notes = [];
            for (let i = 0; i < this.inputNotes.length; i++) {
                notes.push({
                    pitch: this.inputNotes[i].pitch,
                    startTime: this.inputNotes[i].startTime,
                    endTime: this.inputNotes[i].startTime + this.inputNotes[i].duration,
                });
            }
            const unquantizedSequence = {
                notes: notes,
                tempos: [{
                        time: 0,
                        qpm: (_a = this.timeSettings) === null || _a === void 0 ? void 0 : _a.bpm
                    }],
                totalTime: 60 / ((_b = this.timeSettings) === null || _b === void 0 ? void 0 : _b.bpm) * ((_c = this.timeSettings) === null || _c === void 0 ? void 0 : _c.numbeats),
            };
            let quantizedSequence = core.sequences.quantizeNoteSequence(unquantizedSequence, 4);
            this.quantizedNotes = quantizedSequence;
        }
        else {
            throw new Error("No input notes!");
        }
    }
    generateNewSequence() {
        return __awaiter(this, void 0, void 0, function* () {
            let midiNotes = this.quantizedNotes.notes.map(n => n.pitch);
            console.log(midiNotes);
            const notes = midiNotes.map(Tonal.Note.fromMidi);
            console.log(notes);
            const possibleChords = Detect.chord(notes);
            console.log(possibleChords);
            try {
                return yield this.model.continueSequence(this.quantizedNotes, 16, 1.2, possibleChords);
            }
            catch (error) {
                console.log(error);
                return;
            }
        });
    }
}
//retrieve time Settings from Max Patch, initialize Improvisor
maxApi.addHandler("setTimeSettings", (numbeats, beatvalue, measuredbeat, isDotted, bpm) => {
    const timeSettings = {
        numbeats: numbeats,
        beatvalue: beatvalue,
        measuredbeat: measuredbeat,
        isDotted: isDotted,
        bpm: bpm
    };
    improvisor = new Improvisor(timeSettings);
    console.log(improvisor.timeSettings);
});
maxApi.addHandler("getNotes", (...midiNotes) => {
    if (midiNotes.length == 0) {
        return;
    }
    let notes = [];
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
        notes.push(note);
    }
    improvisor.inputNotes = notes;
    console.log(improvisor.inputNotes);
    improvisor.quantizeInputNotes();
    console.log(improvisor.quantizedNotes);
});
maxApi.addHandler("generateSequence", () => __awaiter(void 0, void 0, void 0, function* () {
    let generated = yield improvisor.generateNewSequence();
    console.log("generated", generated);
}));
// 	// const notes = midiNotes.map(Tonal.Note.fromMidi);
// 	// const possibleChords = Detect.chord(notes);
//     // console.log(noteEventList)
// });
