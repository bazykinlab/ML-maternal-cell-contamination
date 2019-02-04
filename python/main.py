import argparse

from preprocessing import VCF
from recalibrator import Recalibrator

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("path", help="path to input VCF file")
	parser.add_argument("sample_name", help="name of sample column in VCF file")
	parser.add_argument("mother_name", help="name of mather column in VCF file")
	parser.add_argument("father_name", help="name of father column in VCF file")


	parser.add_argument("--mode", help="mode", choices=["recalibrate",
														"contamination"],
											   default="recalibrate")
	parser.add_argument("--model_path", help="path to saved recalibrator model", default="../model.pickle")
	parser.add_argument("--method", help="recalibration method",
									choices=["xgboost", "logistic-regression", "confidence-intervals", "meta-classifier"],
									default="xgboost")
	parser.add_argument("--contamination", help="provided contamination estimate")
	parser.add_argument("--output", help="path to output VCF file")


	args = parser.parse_args()

	vcf = VCF(args.path)

	if args.contamination:
		vcf.process(args.sample_name,
					args.mother_name,
					args.father_name,
					args.contamination)

	else:
		vcf.process(args.sample_name,
					args.mother_name,
					args.father_name)

	if args.mode == "contamination":
		print(vcf.estimated_contamination)

	else:
		if not args.output:
			args.output = ".".join(args.path.split(".")[:-1]) + "_recalibrated.vcf"
		print(args.output)

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