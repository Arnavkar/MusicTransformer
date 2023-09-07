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
//require("@tensorflow/tfjs-node");
//import { zeros } from "@tensorflow/tfjs-node";
const utils_1 = require("./utils");
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
        var _a, _b, _c;
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
                    tempos: [{
                            time: 0,
                            qpm: (_a = this.timeSettings) === null || _a === void 0 ? void 0 : _a.qpm
                        }],
                    totalTime: 60 / ((_b = this.timeSettings) === null || _b === void 0 ? void 0 : _b.qpm) * ((_c = this.timeSettings) === null || _c === void 0 ? void 0 : _c.numbeats),
                };
                let quantizedSequence = core.sequences.quantizeNoteSequence(unquantizedSequence, 4);
                this.quantizedNotes = quantizedSequence;
            }
            else {
                throw new Error("input notes could not be Quantized!");
            }
        }
        catch (error) {
            (0, utils_1.logErrorToMax)(error);
        }
    }
    generateNewSequence() {
        return __awaiter(this, void 0, void 0, function* () {
            let midiNotes = this.quantizedNotes.notes.map(n => n.pitch);
            //console.log(midiNotes);
            const notes = midiNotes.map(Tonal.Note.fromMidi);
            //console.log(notes);
            const possibleChords = Detect.chord(notes);
            //console.log(possibleChords);
            try {
                //Provide quantized note sequence, steps to generate, temperature and chord progression
                return yield this.model.continueSequence(this.quantizedNotes, 16, 1.2, possibleChords);
            }
            catch (error) {
                (0, utils_1.logErrorToMax)(error);
                //return empty seqeunce
                return core.sequences.createQuantizedNoteSequence();
            }
        });
    }
}
exports.Improvisor = Improvisor;
