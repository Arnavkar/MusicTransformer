import { Improvisor } from "./improvisor";
import { NumBeats, BeatValue, StepsPerQuarter, TimeSettings, Note} from "./utils";

const maxApi = require("max-api");
let improvisor: Improvisor;
//retrieve time Settings from Max Patch, initialize Improvisor
maxApi.addHandler("setTimeSettings", (numbeats:NumBeats, beatvalue:BeatValue, stepsPerQuarter:StepsPerQuarter, qpm:number) => {
    //measured beat is currently not used
    const timeSettings:TimeSettings = {
        numbeats: numbeats,
        beatvalue: beatvalue,
        stepsPerQuarter: stepsPerQuarter,
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
    console.log("INPUT NOTES–––––––––––––\n",improvisor.inputNotes)
    improvisor.quantizeInputNotes();
    console.log("QUANTIZED INPUT NOTES–––––––––––––\n", improvisor.quantizedInput);
});

maxApi.addHandler("generateSequence", async() => {
    let generated = await improvisor.generateNewSequence();
    console.log("GENERATED NOTES–––––––––––––\n", generated)
    maxApi.outlet(generated);
})