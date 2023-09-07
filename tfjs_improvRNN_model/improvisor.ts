//REMEMBER TO activate py27 environment

import * as core from "@magenta/music/node/core";
import { MusicRNN } from "@magenta/music/node/music_rnn";

const _ = require('lodash');
const Detect = require('tonal-detect');
const Tonal = require('tonal')

//require("@tensorflow/tfjs-node");

//import { zeros } from "@tensorflow/tfjs-node";
import { TimeSettings, ChordProg, NumBeats, BeatValue, StepsPerQuarter, Note, logErrorToMax} from "./utils"
import type { NoteSequence } from "@magenta/music/node/protobuf/index.d.ts";
import { tensorflow } from "@magenta/music/node/protobuf/proto";

export class Improvisor {
    model: MusicRNN;
    timeSettings: TimeSettings;
    currentChordProg: ChordProg;
    inputNotes: Note[];
    quantizedNotes: NoteSequence;

    constructor(timeSettings: TimeSettings) {
        this.model = this.loadModel('https://storage.googleapis.com/download.magenta.tensorflow.org/tfjs_checkpoints/music_rnn/chord_pitches_improv')
        this.timeSettings = timeSettings;
        this.currentChordProg = [];
        this.inputNotes = [];
        this.quantizedNotes = core.sequences.createQuantizedNoteSequence();
    }

    loadModel(path:string):MusicRNN {
        // Using the Improv RNN pretrained model from https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn
        let rnn = new MusicRNN(path);
        rnn.initialize();
        return rnn;
    }

    updateTimeSettings(timeSettings: TimeSettings) {
        this.timeSettings = timeSettings;
    }

    updateChordProg(chordProg: ChordProg) {
        this.currentChordProg = chordProg;
    }

    quantizeInputNotes(){
        try{
            if (!this.model){ throw new Error("Model not loaded!"); }
            if (!this.model.isInitialized){ throw new Error("Model not Initialized!");}
            if (!this.timeSettings){ throw new Error("Time Settings not initialized!");}
            if (!this.inputNotes){ throw new Error("No input notes!"); }

            if (this.inputNotes){
                let notes = [];
                for (let i = 0; i < this.inputNotes.length; i++){
                    notes.push({
                        pitch: this.inputNotes[i].pitch,
                        startTime: this.inputNotes[i].startTime,
                        endTime: this.inputNotes[i].startTime + this.inputNotes[i].duration,
                    });
                }

                const unquantizedSequence = {
                    notes:notes,
                    tempos: [{
                        time: 0, 
                        qpm: this.timeSettings?.qpm
                    }],
                    totalTime: 60 / this.timeSettings?.qpm * this.timeSettings?.numbeats,
                }

                let quantizedSequence = core.sequences.quantizeNoteSequence(unquantizedSequence, 4)
                this.quantizedNotes = quantizedSequence;
            } else {
                throw new Error("input notes could not be Quantized!");
            }
        } catch (error : any) {
            logErrorToMax(error);
        }
    }

    async generateNewSequence(): Promise<tensorflow.magenta.INoteSequence > {
        let midiNotes = this.quantizedNotes.notes.map(n => n.pitch);
        //console.log(midiNotes);
        const notes = midiNotes.map(Tonal.Note.fromMidi);
        //console.log(notes);
	    const possibleChords:ChordProg = Detect.chord(notes);
        //console.log(possibleChords);

        try {
            //Provide quantized note sequence, steps to generate, temperature and chord progression
            return await this.model.continueSequence(this.quantizedNotes, 16, 1.2, possibleChords);
        } catch (error: any) {
            logErrorToMax(error);
            //return empty seqeunce
            return core.sequences.createQuantizedNoteSequence();
        }
    }
}










