import argparse
import logging

from preprocessing import VCF
from recalibrator import Recalibrator

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("path", help="path to input VCF file")
	parser.add_argument("sample_name", help="name of sample column in VCF file")
	parser.add_argument("mother_name", help="name of mather column in VCF file")
	parser.add_argument("father_name", help="name of father column in VCF file")

	parser.add_argument('-v', '--verbose', action='count', default=0, help="Verbose mode")
	parser.add_argument('-r', '--recalibrate', action='count', default=0, help="Recalibrate a VCF file")
	parser.add_argument('-c', '--estimate_contamination', action='count', default=0, help="Estimate the fraction of maternal cell contamination")

	parser.add_argument("--model_path", help="path to saved recalibrator model", default="../model.pickle")
	parser.add_argument("--method", help="recalibration method",
									choices=["xgboost", "logistic-regression", "confidence-intervals", "meta-classifier"],
									default="meta-classifier")
	parser.add_argument("--contamination", help="provided contamination estimate")
	parser.add_argument("--output", help="path to output VCF file")

	parser.add_argument("--GQ_sa", help="Genotype quality cutoff to use when estimating contamination (sample column)", type=int, default=-1)
	parser.add_argument("--GQ_mo", help="Genotype quality cutoff to use when estimating contamination (mother column)", type=int, default=30)
	parser.add_argument("--GQ_fa", help="Genotype quality cutoff to use when estimating contamination (father column)", type=int, default=30)
	parser.add_argument("--DP_sa", help="Read depth cutoff to use when estimating contamination (sample column)", type=int, default=10)
	parser.add_argument("--DP_mo", help="Read depth cutoff to use when estimating contamination (mother column)", type=int, default=10)
	parser.add_argument("--DP_fa", help="Read depth cutoff to use when estimating contamination (father column)", type=int, default=10)
	parser.add_argument("--mode", help="Averaging mode for contamination estimation", choices=['micro', 'macro'], default='micro')

	args = parser.parse_args()

	vcf = VCF(args.path)

	if args.verbose == 0:
		logging.basicConfig(level=logging.WARNING)

	if args.verbose > 0:
		logging.basicConfig(level=logging.INFO)

	param_dict = {
		"GQ_sa": args.GQ_sa,
		"GQ_mo": args.GQ_mo,
		"GQ_fa": args.GQ_fa,
		"DP_sa": args.DP_sa,
		"DP_mo": args.DP_mo,
		"DP_fa": args.DP_fa,
		"mode": args.mode
	}

	if args.contamination:
		vcf.process(args.sample_name,
					args.mother_name,
					args.father_name,
					args.contamination,
					param_dict=param_dict)

	else:
		vcf.process(args.sample_name,
					args.mother_name,
					args.father_name,
					param_dict=param_dict)

	if args.estimate_contamination:
		logging.info("Estimating maternal cell contamination fraction...")
		print(vcf.estimated_contamination)

	if args.recalibrate:
		if not args.output:
			args.output = ".".join(args.path.split(".")[:-1]) + "_recalibrated.vcf"
		logging.info("Output path: {}".format(args.output))

		r = Recalibrator()
		r.load(args.model_path)

		if args.method == 'xgboost':
			preds = r.model_xgb.predict(vcf.prepare_input())

		elif args.method == 'logistic-regression':
			preds = r.model_lr.predict(vcf.prepare_input())

		elif args.method == 'confidence-intervals':
			preds = r.model_ci.predict(vcf.prepare_input())

		elif args.method == 'meta-classifier':
			preds = r.model_meta.predict(vcf.prepare_input())

		vcf.save_predictions(preds, filename=args.output, sample=args.sample_name)

	if not (args.estimate_contamination or args.recalibrate):
		logging.warning("Specify whether to recalibrate, estimate contamination, or both with -r and -c")