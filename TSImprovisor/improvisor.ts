//REMEMBER TO activate py27 environment

import * as core from "@magenta/music/node/core";
import { MusicRNN } from "@magenta/music/node/music_rnn";

const _ = require('lodash');
const Detect = require('tonal-detect');
const Tonal = require('tonal')
//require("@tensorflow/tfjs-node");

//import { zeros } from "@tensorflow/tfjs-node";
import { TimeSettings, ChordProg, NumBeats, BeatValue, StepsPerQuarter, Note, logErrorToMax } from "./utils"
import type { NoteSequence } from "@magenta/music/node/protobuf/index.d.ts";
import { tensorflow } from "@magenta/music/node/protobuf/proto";

export class Improvisor {
    model: MusicRNN;
    timeSettings: TimeSettings;
    currentChordProg: ChordProg;
    inputNotes: Note[];
    quantizedInput: NoteSequence;

    constructor(timeSettings: TimeSettings) {
        this.model = this.loadModel('https://storage.googleapis.com/download.magenta.tensorflow.org/tfjs_checkpoints/music_rnn/chord_pitches_improv')
        this.timeSettings = timeSettings;
        this.currentChordProg = [];
        this.inputNotes = [];
        this.quantizedInput = core.sequences.createQuantizedNoteSequence(
            this.timeSettings.stepsPerQuarter,
            this.timeSettings.qpm
        );
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
                    tempos: [
                        {
                        time: 0, 
                        qpm: this.timeSettings.qpm
                        }
                    ],
                    totalTime: 60 / this.timeSettings.qpm * this.timeSettings.numbeats,
                }

                let quantizedSequence = core.sequences.quantizeNoteSequence(unquantizedSequence, this.timeSettings.stepsPerQuarter);
                this.quantizedInput = quantizedSequence;
            } else {
                throw new Error("input notes could not be Quantized!");
            }
        } catch (error : any) {
            console.log(error);
        }3
    }

    async generateNewSequence(): Promise<tensorflow.magenta.INoteSequence > {
        let midiNotes = this.quantizedInput.notes.map(n => n.pitch);
        console.log("midinotes",midiNotes);
        const notes = midiNotes.map(Tonal.Note.fromMidi);
        console.log("notes",notes);
	    const possibleChords:ChordProg = Detect.chord(notes);
        console.log("chords",possibleChords);

        let stepsToGenerate = this.getNumStepsToGenerate();
        try {
            //Provide quantized note sequence, steps to generate, temperature and chord progression
            return await this.model.continueSequence(
                this.quantizedInput,
                stepsToGenerate,                
                1.2,
                possibleChords
            );
        } catch (error: any) {
            console.log(error);
            //return empty seqeunce
            return core.sequences.createQuantizedNoteSequence(
                this.timeSettings.stepsPerQuarter,
                this.timeSettings.qpm
            );
        }
    }
    
    //Necessary because tempo is in quarter notes per minute
    /* 
    eg.
    3/4 with 12 steps per quarter = 36
    6/8 has 6 beats, so each must have only 6 steps per quarter to have 36 steps
    */
    getNumStepsToGenerate(): number {
        let numSteps: number = this.timeSettings.numbeats * this.timeSettings.stepsPerQuarter;
        if(this.timeSettings.beatvalue == 8) numSteps /= 2;
        return numSteps;
    }
}










