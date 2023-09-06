export type NumBeats = 2 | 3 | 4 | 5 | 6 | 7 | 8;
export type BeatValue = 4 | 8;
export type BeatDivision = 1 | 2 | 4 | 8 | 16 | 32;
//Array of strings representing chords
export type ChordProg = Array<string>;

export interface Note {
    pitch: number;
    velocity: number;
    duration: number;
    startTime: number; //start time from start of the bar
}

export interface TimeSettings {
    numbeats: NumBeats
    beatvalue: BeatValue
    measuredbeat:BeatDivision
    isDotted:boolean
    qpm:number //quarter notes per minute
}