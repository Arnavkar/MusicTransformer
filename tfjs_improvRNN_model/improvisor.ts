//REMEMBER TO activate py27 environment

import * as core from "@magenta/music/node/core";
import { MusicRNN } from "@magenta/music/node/music_rnn";

const _ = require('lodash');
const Detect = require('tonal-detect');
const Tonal = require('tonal')
const maxApi = require("max-api");
//require("@tensorflow/tfjs-node");

//import { zeros } from "@tensorflow/tfjs-node";
import {TimeSettings, ChordProg, NumBeats, BeatValue, BeatDivision, Note} from "./utils"
import type { NoteSequence } from "@magenta/music/node/protobuf/index.d.ts";
import { tensorflow } from "@magenta/music/node/protobuf/proto";

let improvisor: Improvisor;

class Improvisor {
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
        } catch (error) {
            maxApi.post(error, maxApi.POST_LEVELS.ERROR);
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
        } catch (error) {
            maxApi.post(error, maxApi.POST_LEVELS.ERROR);
            //return empty seqeunce
            return core.sequences.createQuantizedNoteSequence();
        }
    }
}

//retrieve time Settings from Max Patch, initialize Improvisor
maxApi.addHandler("setTimeSettings", (numbeats:NumBeats, beatvalue:BeatValue, measuredbeat:BeatDivision, isDotted:boolean, qpm:number) => {
    //measured beat is currently not used
    const timeSettings:TimeSettings = {
        numbeats: numbeats,
        beatvalue: beatvalue,
        measuredbeat: measuredbeat,
        isDotted: isDotted,
        qpm: qpm
    }
    if (improvisor) {
        improvisor.updateTimeSettings(timeSettings);
    } else {
        improvisor = new Improvisor(timeSettings);
    }
    console.log(improvisor.timeSettings);
});

maxApi.addHandler("getNotes", (...midiNotes: number[]) => { //need to use spread operator to grab all values
    if (midiNotes.length == 0){
        return
    } 
    let notes = [];
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
        notes.push(note);
    }
    improvisor.inputNotes = notes;
    //console.log(improvisor.inputNotes)
    improvisor.quantizeInputNotes();
    //console.log(improvisor.quantizedNotes);
});

maxApi.addHandler("generateSequence", async() => {
    let generated = await improvisor.generateNewSequence();
    console.log("generated", generated);
})










