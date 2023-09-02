//REMEMBER TO activate py27 environment

const core = require('@magenta/music/node/core');
const mm = require('@magenta/music/node/music_rnn');
//const Tone = require('tone')
//require('@tensorflow/tfjs-node');
const Tonal = require('tonal')
const Detect = require("tonal-detect")
const _ = require('lodash');

const easymidi = require('easymidi');
let inputs = easymidi.getInputs();
//input = from max 1
let input = new easymidi.Input(inputs[0]);

input.on('noteon', function (msg) {
  console.log(msg)
});

// async function toNoteSequence(input) {
// 	let notes = [];
// 	for (let i = 0; i < input.length; i++) {
// 	  if (input[i] === -1 && notes.length) {
// 		_.last(notes).endTime = i * 0.5;
// 	  } else if (input[i] !== -2 && input[i] !== -1) {
// 		if (notes.length && !_.last(notes).endTime) {
// 		  _.last(notes).endTime = i * 0.5;
// 		}
// 		notes.push({
// 		  pitch: input[i],
// 		  startTime: i * 0.5
// 		});
// 	  }
// 	}
// 	if (notes.length && !_.last(notes).endTime) {
// 	  _.last(notes).endTime = input.length * 0.5;
// 	}
// }
// input.on('noteoff', function (msg) {
//     console.log(msg)
// });