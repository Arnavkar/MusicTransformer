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
exports.Improvisor = void 0;
const core = __importStar(require("@magenta/music/node/core"));
const music_rnn_1 = require("@magenta/music/node/music_rnn");
const _ = require('lodash');
const Detect = require('tonal-detect');
const Tonal = require('tonal');
class Improvisor {
    constructor(timeSettings) {
        this.model = this.loadModel('https://storage.googleapis.com/download.magenta.tensorflow.org/tfjs_checkpoints/music_rnn/chord_pitches_improv');
        this.timeSettings = timeSettings;
        this.currentChordProg = [];
        this.inputNotes = [];
        this.quantizedInput = core.sequences.createQuantizedNoteSequence(this.timeSettings.stepsPerQuarter, this.timeSettings.qpm);
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
        try {
            if (!this.model) {
                throw new Error("Model not loaded!");
            }
            if (!this.model.isInitialized) {
                throw new Error("Model not Initialized!");
            }
            if (!this.timeSettings) {
                throw new Error("Time Settings not initialized!");
            }
            if (!this.inputNotes) {
                throw new Error("No input notes!");
            }
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
                    tempos: [
                        {
                            time: 0,
                            qpm: this.timeSettings.qpm
                        }
                    ],
                    totalTime: 60 / this.timeSettings.qpm * this.timeSettings.numbeats,
                };
                let quantizedSequence = core.sequences.quantizeNoteSequence(unquantizedSequence, this.timeSettings.stepsPerQuarter);
                this.quantizedInput = quantizedSequence;
            }
            else {
                throw new Error("input notes could not be Quantized!");
            }
        }
        catch (error) {
            console.log(error);
        }
        3;
    }
    generateNewSequence() {
        return __awaiter(this, void 0, void 0, function* () {
            let midiNotes = this.quantizedInput.notes.map(n => n.pitch);
            console.log("midinotes", midiNotes);
            const notes = midiNotes.map(Tonal.Note.fromMidi);
            console.log("notes", notes);
            const possibleChords = Detect.chord(notes);
            console.log("chords", possibleChords);
            let stepsToGenerate = this.getNumStepsToGenerate();
            try {
                //Provide quantized note sequence, steps to generate, temperature and chord progression
                return yield this.model.continueSequence(this.quantizedInput, stepsToGenerate, 1.2, possibleChords);
            }
            catch (error) {
                console.log(error);
                //return empty seqeunce
                return core.sequences.createQuantizedNoteSequence(this.timeSettings.stepsPerQuarter, this.timeSettings.qpm);
            }
        });
    }
    //Necessary because tempo is in quarter notes per minute
    /*
    eg.
    3/4 with 12 steps per quarter = 36
    6/8 has 6 beats, so each must have only 6 steps per quarter to have 36 steps
    */
    getNumStepsToGenerate() {
        let numSteps = this.timeSettings.numbeats * this.timeSettings.stepsPerQuarter;
        if (this.timeSettings.beatvalue == 8)
            numSteps /= 2;
        return numSteps;
    }
}
exports.Improvisor = Improvisor;
